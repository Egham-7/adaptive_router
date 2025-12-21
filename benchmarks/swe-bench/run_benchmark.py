#!/usr/bin/env python3
"""
SWE-bench Verified benchmarking for Adaptive AI router.

This script generates predictions using the Adaptive routing model
(Claude Opus 4.5 + Claude Sonnet 4.5), submits them to SWE-bench
for cloud-based evaluation, and tracks detailed metrics.

SWE-bench Verified (500 instances) is used for leaderboard eligibility.
Dataset loaded directly from HuggingFace: SWE-bench/SWE-bench_Verified

Usage:
    python run_benchmark.py --quick      # Test with 5 instances
    python run_benchmark.py --medium     # Test with 50 instances
    python run_benchmark.py --full       # All 500 Verified instances (leaderboard)
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import get_adaptive_config, get_swebench_config, settings
from src.models import AdaptiveModel
from src.models.base import SWEBenchInstanceMetrics
from src.utils import ResultTracker
from src.utils.dataset_loader import get_dataset_instances
from src.utils.swebench_integration import SWEBenchClient, create_predictions_file

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_predictions(
    model: AdaptiveModel,
    instances: list[dict],
    temperature: float,
    max_tokens: int,
) -> tuple[list[dict], ResultTracker]:
    """
    Generate patch predictions for all instances.

    Args:
        model: Adaptive model to use
        instances: List of SWE-bench instances
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Tuple of (predictions_list, result_tracker)
    """
    predictions = []
    tracker = ResultTracker(model_name=model.get_model_name(), dataset="swe-bench_verified")

    logger.info(f"\n{'='*70}")
    logger.info(f"Generating predictions for {len(instances)} instances")
    logger.info(f"{'='*70}\n")

    for i, instance in enumerate(instances, 1):
        instance_id = instance["instance_id"]
        logger.info(f"[{i}/{len(instances)}] Processing {instance_id}")

        try:
            # Generate patch
            metrics = model.solve_instance(
                instance_id=instance_id,
                repo=instance.get("repo", "unknown"),
                base_commit=instance.get("base_commit", ""),
                problem_statement=instance["problem_statement"],
                repo_context="",  # Could add context here
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Add to tracker
            tracker.add_instance_result(metrics)

            # Add to predictions if patch was generated
            if metrics.patch_generated and metrics.patch_content:
                predictions.append({
                    "instance_id": instance_id,
                    "model_name_or_path": "adaptive-router",
                    "model_patch": metrics.patch_content,
                })

                logger.info(
                    f"  ✓ Patch generated | "
                    f"Cost: ${metrics.total_cost:.4f} | "
                    f"Tokens: {metrics.total_tokens} | "
                    f"Model: {metrics.generation_metrics.model_used}"
                )
            else:
                logger.warning("  ✗ Failed to generate patch")
                predictions.append({
                    "instance_id": instance_id,
                    "model_name_or_path": "adaptive-router",
                    "model_patch": "",
                })

        except Exception as e:
            logger.error(f"  ✗ Error: {str(e)}")
            predictions.append({
                "instance_id": instance_id,
                "model_name_or_path": "adaptive-router",
                "model_patch": "",
            })

    # Add model selection stats
    tracker.set_model_selection_stats(model.get_model_selection_stats())

    logger.info(f"\n{'='*70}")
    logger.info(f"Generation complete: {len(predictions)} predictions")
    logger.info(f"{'='*70}\n")

    return predictions, tracker


async def generate_predictions_async(
    model: AdaptiveModel, instances: list[dict], temperature: float, max_tokens: int
) -> tuple[list[dict], ResultTracker]:
    """
    Generate predictions asynchronously for all instances using Adaptive routing.

    Args:
        model: AdaptiveModel instance
        instances: List of SWE-bench instances
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Tuple of (predictions list, ResultTracker)
    """
    predictions = []
    tracker = ResultTracker(model_name=model.get_model_name(), dataset="swe-bench_verified")

    logger.info(f"\n{'='*70}")
    logger.info(f"Generating predictions for {len(instances)} instances (async)")
    logger.info(f"{'='*70}\n")

    # Create all tasks
    tasks = [
        model.solve_instance_async(
            instance_id=inst["instance_id"],
            repo=inst.get("repo", "unknown"),
            base_commit=inst.get("base_commit", ""),
            problem_statement=inst["problem_statement"],
            repo_context="",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for inst in instances
    ]

    # Run all tasks concurrently and wait for results
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for i, (instance, result) in enumerate(zip(instances, results), 1):
        instance_id = instance["instance_id"]
        logger.info(f"[{i}/{len(instances)}] Processing {instance_id}")

        if isinstance(result, Exception):
            logger.error(f"  ✗ Error: {str(result)}")
            predictions.append({
                "instance_id": instance_id,
                "model_name_or_path": "adaptive-router",
                "model_patch": "",
            })
        elif isinstance(result, SWEBenchInstanceMetrics):
            # Add to tracker
            tracker.add_instance_result(result)

            # Add to predictions
            if result.patch_generated and result.patch_content:
                predictions.append({
                    "instance_id": instance_id,
                    "model_name_or_path": "adaptive-router",
                    "model_patch": result.patch_content,
                })
                logger.info(
                    f"  ✓ Patch generated | "
                    f"Cost: ${result.total_cost:.4f} | "
                    f"Tokens: {result.total_tokens} | "
                    f"Model: {result.generation_metrics.model_used}"
                )
            else:
                logger.warning("  ✗ Failed to generate patch")
                predictions.append({
                    "instance_id": instance_id,
                    "model_name_or_path": "adaptive-router",
                    "model_patch": "",
                })

    # Add model selection stats
    tracker.set_model_selection_stats(model.get_model_selection_stats())

    logger.info(f"\n{'='*70}")
    logger.info(f"Generation complete: {len(predictions)} predictions")
    logger.info(f"{'='*70}\n")

    return predictions, tracker


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run SWE-bench Lite benchmarks with Adaptive AI router"
    )

    # Benchmark mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true", help="Quick test with 5 instances")
    mode_group.add_argument("--medium", action="store_true", help="Medium test with 50 instances")
    mode_group.add_argument(
        "--full", action="store_true", help="Full benchmark with all 500 Verified instances (leaderboard)"
    )

    # Configuration options
    parser.add_argument("--max-instances", type=int, help="Maximum number of instances to test")
    parser.add_argument("--temperature", type=float, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument(
        "--cost-bias", type=float, help="Adaptive cost bias (0.0=best performance, 1.0=cheapest)"
    )
    parser.add_argument("--run-id", type=str, help="Custom run ID (default: adaptive_TIMESTAMP)")
    parser.add_argument(
        "--skip-submit",
        action="store_true",
        help="Skip submission to SWE-bench (only generate predictions)",
    )
    parser.add_argument(
        "--wait", action="store_true", help="Wait for evaluation results before exiting"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        swebench_config = get_swebench_config()
        adaptive_config = get_adaptive_config()

        # Apply CLI overrides
        if args.max_instances is not None:
            swebench_config.max_instances = args.max_instances
        if args.temperature is not None:
            swebench_config.temperature = args.temperature
        if args.max_tokens is not None:
            swebench_config.max_tokens = args.max_tokens
        if args.cost_bias is not None:
            adaptive_config.cost_bias = args.cost_bias

        # Quick/medium/full mode overrides
        if args.quick:
            swebench_config.max_instances = 5
            logger.info("Quick mode: Testing with 5 instances")
        elif args.medium:
            swebench_config.max_instances = 50
            logger.info("Medium mode: Testing with 50 instances")
        elif args.full:
            swebench_config.max_instances = 500  # All Verified instances
            logger.info("Full mode: Testing with all 500 Verified instances (leaderboard run)")

        # Print configuration
        settings.print_summary()

        # Validate configuration
        settings.validate()

        # Initialize Adaptive model
        logger.info("Initializing Adaptive model...")
        model = AdaptiveModel(
            api_key=adaptive_config.api_key,
            api_base=adaptive_config.api_base,
            models=adaptive_config.models,
            cost_bias=adaptive_config.cost_bias,
            max_concurrent=adaptive_config.max_concurrent,
            max_requests_per_second=adaptive_config.max_requests_per_second,
        )

        # Load dataset instances
        logger.info(f"Loading SWE-bench {swebench_config.dataset} dataset...")
        instances = get_dataset_instances(
            dataset=swebench_config.dataset, max_instances=swebench_config.max_instances
        )

        # Generate predictions (async)
        predictions, tracker = asyncio.run(
            generate_predictions_async(
                model=model,
                instances=instances,
                temperature=swebench_config.temperature,
                max_tokens=swebench_config.max_tokens,
            )
        )

        # Print generation summary
        tracker.print_summary()

        # Save predictions file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = args.run_id or f"adaptive_{timestamp}"
        predictions_dir = Path("predictions")
        predictions_dir.mkdir(exist_ok=True)
        predictions_path = predictions_dir / f"{run_id}.json"

        create_predictions_file(predictions, predictions_path)

        # Save detailed results
        results_dir = Path("results") / "adaptive"
        results_dir.mkdir(parents=True, exist_ok=True)
        tracker.save_json(results_dir / f"{run_id}_generation.json")
        tracker.save_csv(results_dir / f"{run_id}_generation.csv")

        logger.info(f"\n✓ Predictions saved to: {predictions_path}")
        logger.info(f"✓ Results saved to: {results_dir}/")

        # Submit to SWE-bench
        if not args.skip_submit:
            logger.info(f"\n{'='*70}")
            logger.info("Submitting to SWE-bench for evaluation...")
            logger.info(f"{'='*70}\n")

            if swebench_config.api_key is None:
                logger.error("SWE-bench API key not configured")
                return 1

            client = SWEBenchClient(api_key=swebench_config.api_key)

            # Test connection first
            if not client.test_connection():
                logger.error("Failed to connect to SWE-bench API")
                return 1

            # Submit predictions
            success = client.submit_predictions(
                dataset=swebench_config.dataset,
                split=swebench_config.split,
                predictions_path=predictions_path,
                run_id=run_id,
            )

            if success:
                logger.info("\n✓ Submitted successfully!")
                logger.info(f"\nRun ID: {run_id}")
                logger.info("\nTo check results later, run:")
                logger.info(
                    f"  sb-cli get-report {swebench_config.dataset} {swebench_config.split} {run_id}"
                )

                # Wait for results if requested
                if args.wait:
                    logger.info("\nWaiting for evaluation results...")
                    report = client.wait_for_results(
                        dataset=swebench_config.dataset,
                        split=swebench_config.split,
                        run_id=run_id,
                        timeout_seconds=3600,  # 1 hour
                        poll_interval=30,
                    )

                    if report:
                        logger.info("\n✓ Evaluation complete!")
                        logger.info(json.dumps(report, indent=2))

                        # Save report
                        report_path = results_dir / f"{run_id}_report.json"
                        with open(report_path, "w") as f:
                            json.dump(report, f, indent=2)
                        logger.info(f"\n✓ Report saved to: {report_path}")
                    else:
                        logger.warning("\nEvaluation timed out or failed")
            else:
                logger.error("Submission failed")
                return 1

        return 0

    except KeyboardInterrupt:
        logger.info("\n\nBenchmark interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
