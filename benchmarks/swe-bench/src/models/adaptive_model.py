"""
Adaptive routing model implementation for SWE-bench benchmarking.

This module provides a wrapper around the Adaptive routing API with detailed
cost and token tracking. Tracks which model was actually selected for each request.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

from aiolimiter import AsyncLimiter
from anthropic import Anthropic, AsyncAnthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..utils.response_parser import parse_adaptive_response
from .base import BaseSWEBenchModel, ResponseMetrics, SWEBenchInstanceMetrics

logger = logging.getLogger(__name__)


class AdaptiveModel(BaseSWEBenchModel):
    """
    Adaptive routing model wrapper with response-based cost tracking.

    This implementation uses the Adaptive API and extracts token usage, cost,
    and selected model information directly from API responses.
    """

    def __init__(
        self,
        model_name: str = "adaptive-router",
        api_key: str | None = None,
        api_base: str | None = None,
        models: list[str] | None = None,
        cost_bias: float = 0.5,
        max_concurrent: int = 10,
        max_requests_per_second: float = 20.0,
    ):
        """
        Initialize Adaptive model.

        Args:
            model_name: Adaptive model identifier (default: adaptive-router)
            api_key: Adaptive API key (defaults to ADAPTIVE_API_KEY env var)
            api_base: Adaptive API base URL (defaults to ADAPTIVE_BASE_URL env var)
            models: List of models to route between (None = use server default)
            cost_bias: Cost vs performance trade-off (0=best performance, 1=cheapest)
            max_concurrent: Maximum concurrent requests (default: 10)
            max_requests_per_second: Rate limit for requests (default: 20.0)
        """
        super().__init__(model_name=model_name)

        # Initialize API credentials
        self.api_key = api_key or os.getenv("ADAPTIVE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Adaptive API key not found. Set ADAPTIVE_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.api_base = api_base or os.getenv("ADAPTIVE_BASE_URL", "https://api.llmadaptive.uk")

        # Initialize Anthropic clients (both sync and async)
        self.client = Anthropic(api_key=self.api_key, base_url=self.api_base)
        self.async_client = AsyncAnthropic(api_key=self.api_key, base_url=self.api_base)

        self.models = models
        self.cost_bias = cost_bias

        # Async concurrency controls
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = AsyncLimiter(max_requests_per_second, 1.0)

        # Track model selection statistics (thread-safe with lock)
        self.model_selection_counts: dict[str, int] = {}
        self._stats_lock = asyncio.Lock()

        if self.models:
            self.logger.info(
                f"Initialized Adaptive router with {len(self.models)} models, "
                f"cost_bias={cost_bias}, max_concurrent={max_concurrent}, "
                f"max_rps={max_requests_per_second}"
            )
        else:
            self.logger.info(
                f"Initialized Adaptive router with server default models, "
                f"cost_bias={cost_bias}, max_concurrent={max_concurrent}, "
                f"max_rps={max_requests_per_second}"
            )

    def generate_patch(
        self, problem_statement: str, repo_context: str, temperature: float, max_tokens: int
    ) -> tuple[str, ResponseMetrics]:
        """
        Generate a patch for the given problem using Adaptive routing.

        Args:
            problem_statement: The issue description
            repo_context: Context about the repository
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (patch_content, response_metrics with selected model)
        """
        start_time = time.time()

        try:
            # Construct the prompt for patch generation
            prompt = self._build_patch_prompt(problem_statement, repo_context)

            # Build model_router config - models as string array
            model_router_config: dict[str, Any] = {
                "cost_bias": self.cost_bias,
            }
            if self.models:
                model_router_config["models"] = self.models

            # Call Adaptive API with Anthropic format:
            # - model="" (empty string) for intelligent routing
            # - model_router and fallback in extra_body (request body)
            # - system must be array of TextBlockParam for Adaptive API
            message = self.client.messages.create(
                model="",
                max_tokens=max_tokens,
                system=[{"type": "text", "text": "You are an expert software engineer. Generate precise code patches to fix issues."}],
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                extra_body={
                    "model_router": model_router_config,
                    "fallback": {"mode": "sequential"},
                },
            )

            # Extract generated patch from response
            patch_content = message.content[0].text if message.content else ""

            # Parse metrics from Anthropic-style response
            input_tokens, output_tokens, cost, selected_model = parse_adaptive_response(
                response_json=message.model_dump(), requested_models=self.models or []
            )

            # Calculate latency
            latency = time.time() - start_time

            # Create metrics object
            metrics = ResponseMetrics(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_seconds=latency,
                model_used=selected_model,
            )

            # Track which model was selected
            self.model_selection_counts[selected_model] = (
                self.model_selection_counts.get(selected_model, 0) + 1
            )

            self.logger.debug(
                f"Adaptive selected {selected_model}: " f"{output_tokens} tokens, cost: ${cost:.6f}"
            )

            return patch_content, metrics

        except Exception as e:
            self.logger.error(f"Error calling Adaptive API: {str(e)}")
            latency = time.time() - start_time
            error_metrics = ResponseMetrics(
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_seconds=latency,
                model_used="error",
                error=str(e),
            )
            return "", error_metrics

    def _build_patch_prompt(self, problem_statement: str, repo_context: str) -> str:
        """
        Build a prompt for patch generation.

        Args:
            problem_statement: The issue description
            repo_context: Context about the repository

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are tasked with fixing a bug in a software repository.

## Problem Statement
{problem_statement}
"""

        if repo_context:
            prompt += f"""
## Repository Context
{repo_context}
"""

        prompt += """
## Instructions
1. Analyze the problem statement carefully
2. Identify the root cause of the issue
3. Generate a precise patch (diff format) that fixes the issue
4. Ensure the patch is minimal and doesn't break existing functionality
5. Output ONLY the patch content in unified diff format

Generate the patch now:
"""

        return prompt

    def get_model_selection_stats(self) -> dict[str, Any]:
        """
        Get statistics about which models were selected.

        Returns:
            Dictionary with model selection counts and percentages
        """
        total = sum(self.model_selection_counts.values())
        if total == 0:
            return {}

        stats: dict[str, Any] = {"total_requests": total, "models": {}}

        for model, count in self.model_selection_counts.items():
            stats["models"][model] = {
                "count": count,
                "percentage": round(count / total * 100, 2),
            }

        return stats

    def clear_stats(self) -> None:
        """Clear model selection statistics."""
        self.model_selection_counts = {}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(Exception),
    )
    async def generate_patch_async(
        self, problem_statement: str, repo_context: str, temperature: float, max_tokens: int
    ) -> tuple[str, ResponseMetrics]:
        """
        Generate a patch asynchronously using Adaptive routing.

        Args:
            problem_statement: The issue description
            repo_context: Context about the repository
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (patch_content, response_metrics with selected model)
        """
        start_time = time.time()

        async with self.rate_limiter:  # Rate limiting
            async with self.semaphore:  # Concurrency control
                try:
                    # Construct the prompt for patch generation
                    prompt = self._build_patch_prompt(problem_statement, repo_context)

                    # Build model_router config - models as string array
                    model_router_config: dict[str, Any] = {
                        "cost_bias": self.cost_bias,
                    }
                    if self.models:
                        model_router_config["models"] = self.models

                    # Call Adaptive API with Anthropic format:
                    # - model="" (empty string) for intelligent routing
                    # - model_router and fallback in extra_body (request body)
                    # - system must be array of TextBlockParam for Adaptive API
                    message = await self.async_client.messages.create(
                        model="",
                        max_tokens=max_tokens,
                        system=[{"type": "text", "text": "You are an expert software engineer. Generate precise code patches to fix issues."}],
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        extra_body={
                            "model_router": model_router_config,
                            "fallback": {"mode": "sequential"},
                        },
                    )

                    # Extract generated patch from response
                    patch_content = message.content[0].text if message.content else ""

                    # Parse metrics from Anthropic-style response
                    input_tokens, output_tokens, cost, selected_model = parse_adaptive_response(
                        response_json=message.model_dump(), requested_models=self.models or []
                    )

                    # Calculate latency
                    latency = time.time() - start_time

                    # Create metrics object
                    metrics = ResponseMetrics(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost,
                        latency_seconds=latency,
                        model_used=selected_model,
                    )

                    # Track which model was selected (thread-safe)
                    async with self._stats_lock:
                        self.model_selection_counts[selected_model] = (
                            self.model_selection_counts.get(selected_model, 0) + 1
                        )

                    self.logger.debug(
                        f"Adaptive selected {selected_model}: "
                        f"{output_tokens} tokens, cost: ${cost:.6f}"
                    )

                    return patch_content, metrics

                except Exception as e:
                    self.logger.error(f"Error calling Adaptive API: {str(e)}")
                    latency = time.time() - start_time
                    error_metrics = ResponseMetrics(
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0.0,
                        latency_seconds=latency,
                        model_used="error",
                        error=str(e),
                    )
                    return "", error_metrics

    async def solve_instance_async(
        self,
        instance_id: str,
        repo: str,
        base_commit: str,
        problem_statement: str,
        repo_context: str = "",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> SWEBenchInstanceMetrics:
        """
        Solve a single SWE-bench instance asynchronously.

        Args:
            instance_id: SWE-bench instance identifier
            repo: Repository name
            base_commit: Base commit hash
            problem_statement: Problem description
            repo_context: Optional repository context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            SWEBenchInstanceMetrics with all execution details
        """
        from .base import SWEBenchInstanceMetrics

        start_time = time.time()

        # Initialize metrics
        metrics = SWEBenchInstanceMetrics(
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            problem_statement=problem_statement,
        )

        try:
            # Generate patch asynchronously
            patch_content, generation_metrics = await self.generate_patch_async(
                problem_statement=problem_statement,
                repo_context=repo_context,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Update metrics
            metrics.patch_generated = bool(patch_content and len(patch_content) > 0)
            metrics.patch_content = patch_content
            metrics.generation_metrics = generation_metrics

            if not metrics.patch_generated:
                metrics.resolution_status = "failed"
                metrics.error_message = "Failed to generate patch"
            else:
                # Patch will be evaluated by SWE-bench CLI
                metrics.resolution_status = "generated"

        except Exception as e:
            self.logger.error(f"Error solving instance {instance_id}: {str(e)}")
            metrics.resolution_status = "error"
            metrics.error_message = str(e)

        # Record execution time
        metrics.execution_time_seconds = time.time() - start_time

        return metrics
