"""
Base model classes and metrics for SWE-bench benchmarking.

This module defines the data structures and interfaces for tracking
SWE-bench task execution metrics and model performance.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ResponseMetrics:
    """
    Metrics for a single API call.

    Captures token usage, cost, latency, and which model was actually used.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    model_used: str = "unknown"
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_seconds": round(self.latency_seconds, 3),
            "model_used": self.model_used,
            "error": self.error,
        }


@dataclass
class SWEBenchInstanceMetrics:
    """
    Metrics for a single SWE-bench instance.

    Captures all information about patch generation and test execution.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str

    # Execution results
    patch_generated: bool = False
    patch_content: str = ""
    test_passed: bool = False
    test_output: str = ""
    resolution_status: str = "failed"  # Options: "resolved", "failed", "error"

    # API metrics
    generation_metrics: ResponseMetrics = field(default_factory=ResponseMetrics)

    # Additional metadata
    error_message: str | None = None
    execution_time_seconds: float = 0.0

    @property
    def total_cost(self) -> float:
        """Total cost for this instance."""
        return self.generation_metrics.cost_usd

    @property
    def total_tokens(self) -> int:
        """Total tokens used for this instance."""
        return self.generation_metrics.total_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement[:200] + "...",  # Truncate
            "patch_generated": self.patch_generated,
            "test_passed": self.test_passed,
            "resolution_status": self.resolution_status,
            "generation_metrics": self.generation_metrics.to_dict(),
            "error_message": self.error_message,
            "execution_time_seconds": round(self.execution_time_seconds, 2),
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
        }


@dataclass
class BenchmarkRun:
    """
    Complete results from a benchmark run.

    Aggregates all instance metrics and provides summary statistics.
    """

    model_name: str
    dataset: str  # "lite", "verified", "full"
    timestamp: str
    total_instances: int
    resolved_instances: int
    failed_instances: int
    error_instances: int

    # Aggregate metrics
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_execution_time_seconds: float = 0.0

    # Per-instance results
    instance_results: list[dict[str, Any]] = field(default_factory=list)

    # Model selection statistics (for Adaptive)
    model_selection_stats: dict[str, Any] | None = None

    @property
    def resolution_rate(self) -> float:
        """Percentage of instances successfully resolved."""
        if self.total_instances == 0:
            return 0.0
        return (self.resolved_instances / self.total_instances) * 100

    @property
    def cost_per_instance(self) -> float:
        """Average cost per instance."""
        if self.total_instances == 0:
            return 0.0
        return self.total_cost_usd / self.total_instances

    @property
    def cost_per_resolved(self) -> float:
        """Average cost per resolved instance."""
        if self.resolved_instances == 0:
            return 0.0
        return self.total_cost_usd / self.resolved_instances

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "model_name": self.model_name,
            "dataset": self.dataset,
            "timestamp": self.timestamp,
            "summary": {
                "total_instances": self.total_instances,
                "resolved_instances": self.resolved_instances,
                "failed_instances": self.failed_instances,
                "error_instances": self.error_instances,
                "resolution_rate_percent": round(self.resolution_rate, 2),
            },
            "cost_metrics": {
                "total_cost_usd": round(self.total_cost_usd, 4),
                "cost_per_instance": round(self.cost_per_instance, 4),
                "cost_per_resolved": round(self.cost_per_resolved, 4),
            },
            "token_metrics": {
                "total_tokens": self.total_tokens,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
            },
            "execution_time": {
                "total_seconds": round(self.total_execution_time_seconds, 2),
                "avg_seconds_per_instance": round(
                    (
                        self.total_execution_time_seconds / self.total_instances
                        if self.total_instances > 0
                        else 0
                    ),
                    2,
                ),
            },
            "instance_results": self.instance_results,
        }

        if self.model_selection_stats:
            result["model_selection_stats"] = self.model_selection_stats

        return result


class BaseSWEBenchModel(ABC):
    """
    Abstract base class for SWE-bench models.

    Defines the interface that all model implementations must follow.
    """

    def __init__(self, model_name: str = "unknown"):
        """
        Initialize base model.

        Args:
            model_name: Model identifier
        """
        self.model_name = model_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def generate_patch(
        self, problem_statement: str, repo_context: str, temperature: float, max_tokens: int
    ) -> tuple[str, ResponseMetrics]:
        """
        Generate a patch for the given problem.

        Args:
            problem_statement: The issue description
            repo_context: Context about the repository
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (patch_content, response_metrics)
        """
        pass

    def get_model_name(self) -> str:
        """Return the model identifier."""
        return self.model_name

    def solve_instance(
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
        Solve a single SWE-bench instance.

        This is the main entry point for solving an instance. It:
        1. Generates a patch using the model
        2. Returns metrics for tracking

        Note: Patch application and test execution is handled separately
        by the benchmark runner using SWE-bench CLI.

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
        start_time = time.time()

        # Initialize metrics
        metrics = SWEBenchInstanceMetrics(
            instance_id=instance_id,
            repo=repo,
            base_commit=base_commit,
            problem_statement=problem_statement,
        )

        try:
            # Generate patch
            patch_content, generation_metrics = self.generate_patch(
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
                # Status will be updated later
                metrics.resolution_status = "pending"

        except Exception as e:
            self.logger.error(f"Error solving instance {instance_id}: {str(e)}")
            metrics.resolution_status = "error"
            metrics.error_message = str(e)

        # Record execution time
        metrics.execution_time_seconds = time.time() - start_time

        return metrics
