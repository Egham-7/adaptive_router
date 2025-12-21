"""
SWE-bench CLI integration for submitting and retrieving evaluation results.

This module provides utilities for interacting with the SWE-bench API via sb-cli.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from sb_cli import app as sb_cli_app

logger = logging.getLogger(__name__)

# Map our dataset names to sb-cli subset names
DATASET_TO_SUBSET = {
    "verified": "swe-bench_verified",
    "lite": "swe-bench_lite",
    "full": "swe-bench-m",
}


def _run_sb_cli(args: list[str]) -> tuple[int, str, str]:
    """
    Run sb-cli command via Python import.

    Args:
        args: Command arguments (without 'sb-cli' prefix)

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    import io
    import sys

    # Capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_stdout = io.StringIO()
    sys.stderr = captured_stderr = io.StringIO()

    return_code = 0
    try:
        sb_cli_app(args)
    except SystemExit as e:
        return_code = e.code if isinstance(e.code, int) else 1
    except Exception as e:
        return_code = 1
        print(str(e), file=sys.stderr)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return return_code, captured_stdout.getvalue(), captured_stderr.getvalue()


class SWEBenchClient:
    """Client for interacting with SWE-bench API via sb-cli."""

    def __init__(self, api_key: str):
        """
        Initialize SWE-bench client.

        Args:
            api_key: SWE-bench API key
        """
        self.api_key = api_key
        # Set environment variable for sb-cli
        os.environ["SWEBENCH_API_KEY"] = api_key

    def _get_subset_name(self, dataset: str) -> str:
        """Convert dataset name to sb-cli subset name."""
        return DATASET_TO_SUBSET.get(dataset, dataset)

    def test_connection(self) -> bool:
        """
        Test connection to SWE-bench API.

        Returns:
            True if connection successful
        """
        try:
            return_code, stdout, stderr = _run_sb_cli(
                ["get-quotas", "--api_key", self.api_key]
            )

            if return_code == 0:
                logger.info("✓ SWE-bench API connection successful")
                logger.info(f"Quotas: {stdout.strip()}")
                return True
            else:
                logger.error(f"Connection failed: {stderr}")
                return False

        except Exception as e:
            logger.error(f"Error testing connection: {str(e)}")
            return False

    def submit_predictions(
        self,
        dataset: str,
        split: str,
        predictions_path: Path,
        run_id: str,
    ) -> bool:
        """
        Submit predictions to SWE-bench for evaluation.

        Args:
            dataset: Dataset name (e.g., "verified", "lite")
            split: Dataset split ("dev" or "test")
            predictions_path: Path to predictions JSONL file
            run_id: Unique identifier for this run

        Returns:
            True if submission successful
        """
        try:
            subset = self._get_subset_name(dataset)
            logger.info("Submitting predictions to SWE-bench...")
            logger.info(f"  Subset: {subset} ({split})")
            logger.info(f"  Run ID: {run_id}")
            logger.info(f"  Predictions: {predictions_path}")

            return_code, stdout, stderr = _run_sb_cli([
                "submit",
                subset,
                split,
                "--predictions_path",
                str(predictions_path),
                "--run_id",
                run_id,
                "--api_key",
                self.api_key,
            ])

            if return_code == 0:
                logger.info("✓ Predictions submitted successfully")
                logger.info(stdout)
                return True
            else:
                logger.error(f"Submission failed: {stderr}")
                logger.error(f"Output: {stdout}")
                return False

        except Exception as e:
            logger.error(f"Error submitting predictions: {str(e)}")
            return False

    def get_report(
        self,
        dataset: str,
        split: str,
        run_id: str,
        output_path: Path | None = None,
    ) -> dict[str, Any] | None:
        """
        Get evaluation report from SWE-bench.

        Args:
            dataset: Dataset name
            split: Dataset split
            run_id: Run identifier
            output_path: Optional path to save report JSON

        Returns:
            Report dictionary or None if failed
        """
        try:
            subset = self._get_subset_name(dataset)
            logger.info(f"Retrieving report for run: {run_id}")

            return_code, stdout, stderr = _run_sb_cli([
                "get-report",
                subset,
                split,
                run_id,
                "--api_key",
                self.api_key,
            ])

            if return_code == 0:
                # Parse JSON output
                report_text = stdout.strip()

                # sb-cli may output status messages before JSON
                # Try to find the JSON part
                json_start = report_text.find("{")
                if json_start >= 0:
                    report_json = report_text[json_start:]
                    report = json.loads(report_json)
                else:
                    # If no JSON found, it might be a status message
                    logger.info(f"Report status: {report_text}")
                    return None

                # Save if requested
                if output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w") as f:
                        json.dump(report, f, indent=2)

                return dict(report)
            else:
                logger.warning(f"Could not retrieve report: {stderr}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing report JSON: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error getting report: {str(e)}")
            return None

    def wait_for_results(
        self,
        dataset: str,
        split: str,
        run_id: str,
        timeout_seconds: int = 3600,
        poll_interval: int = 30,
    ) -> dict[str, Any] | None:
        """
        Wait for evaluation results to be ready.

        Args:
            dataset: Dataset name
            split: Dataset split
            run_id: Run identifier
            timeout_seconds: Maximum time to wait
            poll_interval: Seconds between status checks

        Returns:
            Report dictionary when ready, or None if timeout
        """
        start_time = time.time()
        attempts = 0

        logger.info(f"Waiting for evaluation results (timeout: {timeout_seconds}s)...")

        while (time.time() - start_time) < timeout_seconds:
            attempts += 1

            # Try to get report
            report = self.get_report(dataset, split, run_id)

            if report:
                # Check if evaluation is complete
                status = report.get("status", "unknown")

                if status == "completed":
                    logger.info(f"✓ Evaluation completed after {attempts} attempts")
                    return report
                elif status in ["failed", "error"]:
                    logger.error(f"Evaluation failed with status: {status}")
                    return None
                else:
                    logger.info(f"  Status: {status} (attempt {attempts})")

            # Wait before next check
            time.sleep(poll_interval)

        logger.error(f"Timeout waiting for results after {timeout_seconds}s")
        return None

    def list_runs(self, dataset: str, split: str) -> list[str]:
        """
        List all runs for a dataset.

        Args:
            dataset: Dataset name
            split: Dataset split

        Returns:
            List of run IDs
        """
        try:
            subset = self._get_subset_name(dataset)
            return_code, stdout, stderr = _run_sb_cli([
                "list-runs",
                subset,
                split,
                "--api_key",
                self.api_key,
            ])

            if return_code == 0:
                # Parse run IDs from output
                runs = []
                for line in stdout.strip().split("\n"):
                    if line.strip():
                        runs.append(line.strip())
                return runs
            else:
                logger.error(f"Failed to list runs: {stderr}")
                return []

        except Exception as e:
            logger.error(f"Error listing runs: {str(e)}")
            return []


def create_predictions_file(predictions: list[dict[str, str]], output_path: Path) -> None:
    """
    Create predictions file in SWE-bench JSON array format.

    Args:
        predictions: List of predictions with instance_id, model_name_or_path, and model_patch
        output_path: Path to save predictions JSON file

    Format (JSON array):
        [
            {"instance_id": "django__django-11099", "model_name_or_path": "adaptive-router", "model_patch": "..."},
            {"instance_id": "django__django-11100", "model_name_or_path": "adaptive-router", "model_patch": "..."}
        ]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"Saved {len(predictions)} predictions to {output_path}")
