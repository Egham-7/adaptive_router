"""Calculate costs from SWE-bench trajectory files.

Since Nordlys routes to different models dynamically, cost is calculated
post-run by reading the model used from each response in trajectory files.

Pricing can be fetched from Nordlys registry or configured via environment:
    NORDLYS_INPUT_COST_PER_M=3.00   # Input cost per million tokens
    NORDLYS_OUTPUT_COST_PER_M=15.00  # Output cost per million tokens
    NORDLYS_CACHE_READ_COST_PER_M=0.30  # Cache read cost per million tokens
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from nordlys_py import Nordlys


@dataclass
class Usage:
    """Token usage from a single API call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    model: str = ""


@dataclass
class CostSummary:
    """Cost summary for a benchmark run."""

    total_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    api_calls: int = 0
    models_used: dict[str, int] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)


def get_env_pricing() -> dict[str, float] | None:
    """Get pricing from environment variables if set.

    Returns:
        Pricing dict if env vars are set, None otherwise
    """
    input_cost = os.environ.get("NORDLYS_INPUT_COST_PER_M")
    output_cost = os.environ.get("NORDLYS_OUTPUT_COST_PER_M")

    if input_cost and output_cost:
        return {
            "input_cost": float(input_cost) / 1_000_000,
            "output_cost": float(output_cost) / 1_000_000,
            "cache_read_cost": float(os.environ.get("NORDLYS_CACHE_READ_COST_PER_M", "0")) / 1_000_000,
            "cache_write_cost": float(os.environ.get("NORDLYS_CACHE_WRITE_COST_PER_M", "0")) / 1_000_000,
        }
    return None


def get_model_pricing(model_name: str, client: Nordlys | None = None) -> dict[str, float]:
    """Fetch pricing for a model.

    Priority order:
    1. Environment variables (NORDLYS_INPUT_COST_PER_M, etc.)
    2. Nordlys registry lookup
    3. Zero pricing (with warning)

    Args:
        model_name: Model name (e.g., 'nordlys/nordlys-code', 'claude-3-5-sonnet')
        client: Optional Nordlys client (created if not provided)

    Returns:
        Dict with pricing per token (input_cost, output_cost, cache_read_cost, cache_write_cost)
    """
    # Check environment variables first
    env_pricing = get_env_pricing()
    if env_pricing:
        return env_pricing

    # Try Nordlys registry
    if client is None:
        try:
            client = Nordlys()
        except Exception:
            client = None

    if client:
        try:
            model = client.registry.model(model_name)
            provider = model.top_provider

            # Convert from per-million to per-token
            pricing = {
                "input_cost": float(model.pricing.prompt_cost or 0) / 1_000_000,
                "output_cost": float(model.pricing.completion_cost or 0) / 1_000_000,
                "cache_read_cost": 0.0,
                "cache_write_cost": 0.0,
            }

            if provider and provider.pricing:
                pricing["cache_read_cost"] = (
                    float(provider.pricing.input_cache_read_cost or 0) / 1_000_000
                )
                pricing["cache_write_cost"] = (
                    float(provider.pricing.input_cache_write_cost or 0) / 1_000_000
                )

            return pricing
        except Exception:
            pass  # Fall through to zero pricing

    # No pricing found
    print(f"Warning: No pricing found for {model_name}, costs will be zero")
    return {
        "input_cost": 0.0,
        "output_cost": 0.0,
        "cache_read_cost": 0.0,
        "cache_write_cost": 0.0,
    }


def extract_usage_from_trajectory(traj_path: Path) -> list[Usage]:
    """Extract token usage from each API call in a trajectory file.

    Args:
        traj_path: Path to trajectory JSON file

    Returns:
        List of Usage objects for each API call
    """
    usages = []

    try:
        with open(traj_path) as f:
            traj = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse {traj_path}: {e}")
        return usages
    except Exception as e:
        print(f"Warning: Could not read {traj_path}: {e}")
        return usages

    for msg in traj.get("messages", []):
        if msg.get("role") != "assistant":
            continue

        extra = msg.get("extra", {})
        response = extra.get("response", {})

        if not response:
            continue

        usage_data = response.get("usage", {})
        model = response.get("model", "unknown")

        # Handle cached tokens - may be in prompt_tokens_details
        cached = 0
        prompt_details = usage_data.get("prompt_tokens_details", {})
        if prompt_details:
            cached = prompt_details.get("cached_tokens", 0) or 0

        usages.append(
            Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0) or 0,
                completion_tokens=usage_data.get("completion_tokens", 0) or 0,
                cached_tokens=cached,
                model=model,
            )
        )

    return usages


def calculate_trajectory_cost(
    traj_path: Path,
    pricing_cache: dict[str, dict[str, float]] | None = None,
    client: Nordlys | None = None,
) -> tuple[float, list[Usage]]:
    """Calculate total cost for a trajectory file.

    Args:
        traj_path: Path to trajectory JSON file
        pricing_cache: Optional cache of model -> pricing dict
        client: Optional Nordlys client

    Returns:
        Tuple of (total_cost, list of Usage objects)
    """
    if pricing_cache is None:
        pricing_cache = {}
    # client can be None - will fall back to zero pricing

    usages = extract_usage_from_trajectory(traj_path)
    total_cost = 0.0

    for usage in usages:
        model = usage.model

        # Get pricing (from cache or fetch)
        if model not in pricing_cache:
            pricing_cache[model] = get_model_pricing(model, client)

        pricing = pricing_cache[model]

        # Calculate cost for this call
        # Cached tokens are typically charged at cache_read_cost
        # Non-cached prompt tokens are charged at input_cost
        non_cached_prompt = usage.prompt_tokens - usage.cached_tokens
        cost = (
            non_cached_prompt * pricing["input_cost"]
            + usage.cached_tokens * pricing["cache_read_cost"]
            + usage.completion_tokens * pricing["output_cost"]
        )
        total_cost += cost

    return total_cost, usages


def calculate_run_cost(results_dir: Path) -> CostSummary:
    """Calculate total cost for a benchmark run.

    Args:
        results_dir: Path to results directory containing trajectory files

    Returns:
        CostSummary with aggregated costs and usage
    """
    summary = CostSummary()
    pricing_cache: dict[str, dict[str, float]] = {}
    cost_by_model: dict[str, float] = {}

    # Find all trajectory files
    traj_files = list(results_dir.glob("**/*.traj.json"))

    if not traj_files:
        print(f"No trajectory files found in {results_dir}")
        return summary

    # Try to create Nordlys client for registry pricing lookup
    client = None
    try:
        client = Nordlys()
    except Exception:
        pass  # Will fall back to zero pricing

    for traj_path in traj_files:
        cost, usages = calculate_trajectory_cost(traj_path, pricing_cache, client)

        summary.total_cost += cost
        summary.api_calls += len(usages)

        for usage in usages:
            summary.total_prompt_tokens += usage.prompt_tokens
            summary.total_completion_tokens += usage.completion_tokens
            summary.total_cached_tokens += usage.cached_tokens

            # Track models used
            model = usage.model
            summary.models_used[model] = summary.models_used.get(model, 0) + 1

            # Calculate cost per model in single pass (avoid O(n²))
            pricing = pricing_cache.get(model, {})
            non_cached = usage.prompt_tokens - usage.cached_tokens
            model_cost = (
                non_cached * pricing.get("input_cost", 0)
                + usage.cached_tokens * pricing.get("cache_read_cost", 0)
                + usage.completion_tokens * pricing.get("output_cost", 0)
            )
            cost_by_model[model] = cost_by_model.get(model, 0) + model_cost

    summary.cost_by_model = cost_by_model
    return summary


def print_cost_summary(summary: CostSummary) -> None:
    """Print a formatted cost summary."""
    print("\n" + "=" * 60)
    print("COST SUMMARY")
    print("=" * 60)
    print(f"Total Cost:        ${summary.total_cost:.4f}")
    print(f"API Calls:         {summary.api_calls}")
    print(f"Prompt Tokens:     {summary.total_prompt_tokens:,}")
    print(f"Completion Tokens: {summary.total_completion_tokens:,}")
    print(f"Cached Tokens:     {summary.total_cached_tokens:,}")

    if summary.models_used:
        print("\nModels Used:")
        for model, count in sorted(summary.models_used.items(), key=lambda x: -x[1]):
            print(f"  {model}: {count} calls")

    if summary.cost_by_model:
        print("\nCost by Model:")
        for model, cost in sorted(summary.cost_by_model.items(), key=lambda x: -x[1]):
            print(f"  {model}: ${cost:.4f}")

    print("=" * 60 + "\n")


def main() -> None:
    """Calculate costs for the most recent results directory."""
    import sys

    # Default to most recent results
    results_base = Path(__file__).parent.parent / "results"

    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        # Find most recent results directory
        result_dirs = [d for d in results_base.iterdir() if d.is_dir()]
        if not result_dirs:
            print(f"No results directories found in {results_base}")
            return
        results_dir = max(result_dirs, key=lambda d: d.stat().st_mtime)

    print(f"Calculating costs for: {results_dir}")
    summary = calculate_run_cost(results_dir)
    print_cost_summary(summary)


if __name__ == "__main__":
    main()
