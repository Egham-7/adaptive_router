"""
Response parsing utilities for extracting metrics from Adaptive API responses.

This module provides functions to extract token counts and costs from the
Adaptive routing API responses.
"""

import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)


class PricingCalculator:
    """
    Fallback pricing calculator for APIs that don't return cost directly.

    This should only be used as a last resort when the API doesn't provide
    cost information. Pricing may change, so direct API costs are preferred.
    """

    # Pricing per 1M tokens (input, output)
    PRICING = {
        # OpenAI models
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        # Claude models
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
        "claude-sonnet-4-5": (3.00, 15.00),
        "claude-3-5-haiku-20241022": (0.80, 4.00),
        # Google models
        "gemini-2.0-flash-exp": (0.075, 0.30),
        "gemini-1.5-pro": (1.25, 5.00),
        # GLM models
        "glm-4.6": (0.60, 2.20),
    }

    @classmethod
    def calculate_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost based on token counts.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # Find matching pricing
        pricing = None
        for model_key, prices in cls.PRICING.items():
            if model_key in model.lower():
                pricing = prices
                break

        if pricing is None:
            logger.warning(f"No pricing found for model: {model}, using defaults")
            pricing = (1.00, 1.00)  # Default fallback

        input_price, output_price = pricing
        cost = (input_tokens * input_price / 1_000_000) + (output_tokens * output_price / 1_000_000)
        return cost


def parse_adaptive_response(
    response_json: dict[str, Any], requested_models: list[str] | None = None
) -> tuple[int, int, float, str]:
    """
    Parse Adaptive routing API response to extract metrics.

    Args:
        response_json: Adaptive API JSON response (OpenAI-compatible format)
        requested_models: List of models in the routing request (optional)

    Returns:
        Tuple of (input_tokens, output_tokens, cost_usd, model_used)
    """
    try:
        # Extract which model was actually selected
        selected_model = response_json.get("model", "unknown")

        # Extract usage information (supports both OpenAI and Anthropic formats)
        usage = response_json.get("usage", {})

        # Anthropic format: input_tokens, output_tokens
        # OpenAI format: prompt_tokens, completion_tokens
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens", 0)

        # Always calculate cost manually using our pricing table
        cost = PricingCalculator.calculate_cost(
            model=selected_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return input_tokens, output_tokens, cost, selected_model

    except Exception as e:
        logger.error(f"Error parsing Adaptive response: {str(e)}")
        return 0, 0, 0.0, "unknown"


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for a text using tiktoken.

    Args:
        text: Text to count tokens for
        model: Model name for encoding selection

    Returns:
        Estimated token count
    """
    try:
        # Map model names to tiktoken encodings
        if "claude" in model.lower():
            encoding = tiktoken.get_encoding("cl100k_base")
        elif "gpt" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gemini" in model.lower():
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Default to cl100k_base
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    except Exception as e:
        logger.warning(f"Error estimating tokens: {str(e)}, using word count")
        # Fallback: rough estimate (1 token ≈ 0.75 words)
        return int(len(text.split()) * 1.33)
