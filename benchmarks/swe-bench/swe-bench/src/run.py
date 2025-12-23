#!/usr/bin/env python3
"""Run SWE-bench with Nordlys Router.

Uses LiteLLM's anthropic/ provider with Nordlys API endpoint for intelligent routing.
Cost is calculated post-run from trajectory files since Nordlys routes to different
models dynamically.

Requirements:
    - ANTHROPIC_API_KEY and ANTHROPIC_API_BASE must be set in mini-swe-agent config:
        mini-extra config set ANTHROPIC_API_KEY "your-nordlys-api-key"
        mini-extra config set ANTHROPIC_API_BASE "https://api.llmadaptive.uk"

Usage:
    # From benchmarks/ directory:
    uv run python swe-bench/swe-bench/src/run.py

    # Quick test (1 instance)
    uv run python swe-bench/swe-bench/src/run.py --slice :1

    # Custom configuration
    uv run python swe-bench/swe-bench/src/run.py --workers 8 --output results/my-run

    # Skip cost calculation
    uv run python swe-bench/swe-bench/src/run.py --skip-pricing --slice :1

    # Calculate costs after run
    uv run python swe-bench/swe-bench/src/pricing.py results/my-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run SWE-bench with Nordlys Router.

    Returns:
        Exit code from mini-swe-agent command
    """
    # Parse our custom arguments
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--skip-pricing", action="store_true", help="Skip cost calculation")
    known_args, remaining_args = parser.parse_known_args()

    # Cost tracking set to ignore_errors since model names are dynamic
    # Real cost is calculated post-run via pricing.py
    os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

    # Build command arguments
    # Model format: anthropic/nordlys/nordlys-code
    # - LiteLLM uses 'anthropic/' as the provider
    # - Strips prefix and sends 'nordlys/nordlys-code' to API
    # - Nordlys routes to optimal model dynamically
    args = remaining_args or [
        "--model",
        "anthropic/nordlys/nordlys-code",
        "--subset",
        "verified",
        "--split",
        "test",
        "--workers",
        "4",
        "--output",
        "results/nordlys-run",
    ]

    # Ensure model is specified
    if "--model" not in args:
        args = ["--model", "anthropic/nordlys/nordlys-code"] + args

    # Run mini-extra swebench command
    cmd = ["mini-extra", "swebench"] + args
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False)

    # Calculate costs after run (unless --skip-pricing is set)
    if result.returncode == 0 and not known_args.skip_pricing:
        # Find output directory from args
        output_dir = None
        for i, arg in enumerate(args):
            if arg == "--output" and i + 1 < len(args):
                output_dir = Path(args[i + 1])
                break

        if output_dir and output_dir.exists():
            print("\nCalculating costs...")
            try:
                from pricing import calculate_run_cost, print_cost_summary

                summary = calculate_run_cost(output_dir)
                print_cost_summary(summary)
            except Exception as e:
                print(f"Warning: Could not calculate costs: {e}")
                print("Run manually: uv run python swe-bench/swe-bench/src/pricing.py")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
