"""
Result tracking and storage for SWE-bench benchmarks.

This module provides utilities for collecting, aggregating, and storing
benchmark results in various formats.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..models.base import BenchmarkRun, SWEBenchInstanceMetrics

logger = logging.getLogger(__name__)


class ResultTracker:
    """
    Tracks and aggregates results from a benchmark run.

    Collects metrics from individual instances and provides methods
    to save results in multiple formats (JSON, CSV).
    """

    def __init__(self, model_name: str, dataset: str):
        """
        Initialize result tracker.

        Args:
            model_name: Name of the model being benchmarked
            dataset: Dataset being used ("lite", "verified", "full")
        """
        self.model_name = model_name
        self.dataset = dataset
        self.instance_results: list[SWEBenchInstanceMetrics] = []
        self.model_selection_stats: dict[str, Any] | None = None

    def add_instance_result(self, metrics: SWEBenchInstanceMetrics) -> None:
        """
        Add a single instance result to the tracker.

        Args:
            metrics: Metrics from solving one instance
        """
        self.instance_results.append(metrics)
        logger.debug(f"Added result for instance {metrics.instance_id}")

    def set_model_selection_stats(self, stats: dict[str, Any]) -> None:
        """
        Set model selection statistics (for Adaptive routing).

        Args:
            stats: Model selection statistics
        """
        self.model_selection_stats = stats

    def finalize(self) -> BenchmarkRun:
        """
        Finalize the benchmark run and compute aggregate statistics.

        Returns:
            BenchmarkRun with all aggregated metrics
        """
        # Count instances by status
        resolved = sum(1 for m in self.instance_results if m.resolution_status == "resolved")
        failed = sum(1 for m in self.instance_results if m.resolution_status == "failed")
        errors = sum(1 for m in self.instance_results if m.resolution_status == "error")

        # Aggregate token metrics
        total_input_tokens = sum(m.generation_metrics.input_tokens for m in self.instance_results)
        total_output_tokens = sum(m.generation_metrics.output_tokens for m in self.instance_results)
        total_tokens = total_input_tokens + total_output_tokens

        # Aggregate cost
        total_cost = sum(m.total_cost for m in self.instance_results)

        # Aggregate execution time
        total_execution_time = sum(m.execution_time_seconds for m in self.instance_results)

        # Convert instance results to dictionaries
        instance_dicts = [m.to_dict() for m in self.instance_results]

        # Create benchmark run
        benchmark_run = BenchmarkRun(
            model_name=self.model_name,
            dataset=self.dataset,
            timestamp=datetime.now().isoformat(),
            total_instances=len(self.instance_results),
            resolved_instances=resolved,
            failed_instances=failed,
            error_instances=errors,
            total_tokens=total_tokens,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=total_cost,
            total_execution_time_seconds=total_execution_time,
            instance_results=instance_dicts,
            model_selection_stats=self.model_selection_stats,
        )

        logger.info(
            f"Finalized benchmark: {resolved}/{len(self.instance_results)} resolved "
            f"({benchmark_run.resolution_rate:.1f}%), "
            f"cost: ${total_cost:.4f}"
        )

        return benchmark_run

    def save_json(self, output_path: Path) -> None:
        """
        Save complete results to JSON file.

        Args:
            output_path: Path to output file
        """
        benchmark_run = self.finalize()

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        with open(output_path, "w") as f:
            json.dump(benchmark_run.to_dict(), f, indent=2)

        logger.info(f"Saved results to {output_path}")

    def save_csv(self, output_path: Path) -> None:
        """
        Save results summary to CSV file.

        Args:
            output_path: Path to output file
        """
        # Prepare data for CSV
        rows = []
        for metrics in self.instance_results:
            rows.append(
                {
                    "instance_id": metrics.instance_id,
                    "repo": metrics.repo,
                    "patch_generated": metrics.patch_generated,
                    "test_passed": metrics.test_passed,
                    "resolution_status": metrics.resolution_status,
                    "input_tokens": metrics.generation_metrics.input_tokens,
                    "output_tokens": metrics.generation_metrics.output_tokens,
                    "total_tokens": metrics.total_tokens,
                    "cost_usd": metrics.total_cost,
                    "execution_time_seconds": metrics.execution_time_seconds,
                    "model_used": metrics.generation_metrics.model_used,
                    "error_message": metrics.error_message,
                }
            )

        # Create DataFrame and save
        df = pd.DataFrame(rows)

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Saved CSV to {output_path}")

    def print_summary(self) -> None:
        """Print a summary of results to console."""
        benchmark_run = self.finalize()

        print("\n" + "=" * 70)
        print("  SWE-bench Benchmark Results")
        print("=" * 70)
        print(f"\n  Model: {self.model_name}")
        print(f"  Dataset: {self.dataset}")
        print(f"  Timestamp: {benchmark_run.timestamp}")

        print("\n  Resolution Results:")
        print(f"    Total instances:      {benchmark_run.total_instances}")
        print(f"    Resolved:             {benchmark_run.resolved_instances}")
        print(f"    Failed:               {benchmark_run.failed_instances}")
        print(f"    Errors:               {benchmark_run.error_instances}")
        print(f"    Resolution rate:      {benchmark_run.resolution_rate:.2f}%")

        print("\n  Cost Metrics:")
        print(f"    Total cost:           ${benchmark_run.total_cost_usd:.4f}")
        print(f"    Cost per instance:    ${benchmark_run.cost_per_instance:.4f}")
        if benchmark_run.resolved_instances > 0:
            print(f"    Cost per resolved:    ${benchmark_run.cost_per_resolved:.4f}")

        print("\n  Token Metrics:")
        print(f"    Total tokens:         {benchmark_run.total_tokens:,}")
        print(f"    Input tokens:         {benchmark_run.total_input_tokens:,}")
        print(f"    Output tokens:        {benchmark_run.total_output_tokens:,}")

        print("\n  Execution Time:")
        print(f"    Total time:           {benchmark_run.total_execution_time_seconds:.2f}s")
        print(
            f"    Avg per instance:     {benchmark_run.total_execution_time_seconds / benchmark_run.total_instances:.2f}s"
        )

        if self.model_selection_stats:
            print("\n  Model Selection (Adaptive):")
            for model, stats in self.model_selection_stats.get("models", {}).items():
                print(f"    {model:30s} {stats['count']:3d} ({stats['percentage']:5.1f}%)")

        print("\n" + "=" * 70 + "\n")
