"""
Dataset loading utilities for SWE-bench.

This module provides functions to load SWE-bench datasets directly from HuggingFace.
Following SWE-bench best practices: https://www.swebench.com/SWE-bench/guides/datasets/
"""

import logging
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Updated HuggingFace paths (as of Jan 2025)
# See: https://www.swebench.com/SWE-bench/guides/datasets/
DATASET_PATHS = {
    "verified": "SWE-bench/SWE-bench_Verified",  # DEFAULT - 500 instances, leaderboard eligible
    "lite": "SWE-bench/SWE-bench_Lite",  # 300 instances
    "full": "SWE-bench/SWE-bench",  # Full dataset
}


def get_dataset_instances(dataset: str = "verified", max_instances: int = -1) -> list[dict[str, Any]]:
    """
    Load dataset instances directly from HuggingFace.

    Uses the official SWE-bench datasets from HuggingFace Hub.
    HuggingFace handles caching automatically (~/.cache/huggingface/).

    Args:
        dataset: Dataset name ("verified", "lite", "full"). Default is "verified" for leaderboard.
        max_instances: Maximum number of instances (-1 for all)

    Returns:
        List of instance dictionaries

    Raises:
        ValueError: If dataset name is not recognized
    """
    if dataset not in DATASET_PATHS:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Valid options: {list(DATASET_PATHS.keys())}"
        )

    dataset_path = DATASET_PATHS[dataset]
    logger.info(f"Loading {dataset} dataset from HuggingFace: {dataset_path}")

    # Load dataset from HuggingFace
    # HuggingFace caches the dataset automatically
    hf_dataset = load_dataset(dataset_path, split="test")

    # Convert to list of dicts
    instances: list[dict[str, Any]] = [dict(item) for item in hf_dataset]

    logger.info(f"Loaded {len(instances)} instances from {dataset_path}")

    # Limit if requested
    if max_instances > 0 and len(instances) > max_instances:
        instances = instances[:max_instances]
        logger.info(f"Limited to {max_instances} instances")

    return instances


def validate_instance(instance: dict[str, Any]) -> bool:
    """
    Validate that an instance has required fields.

    Args:
        instance: Instance dictionary

    Returns:
        True if valid
    """
    required_fields = [
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
    ]

    for field in required_fields:
        if field not in instance:
            logger.error(f"Instance missing required field: {field}")
            return False

    return True


def get_instance_summary(instances: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Get summary statistics about a dataset.

    Args:
        instances: List of instances

    Returns:
        Dictionary with summary stats
    """
    repos = set()
    avg_problem_length: float = 0

    for instance in instances:
        if "repo" in instance:
            repos.add(instance["repo"])
        if "problem_statement" in instance:
            avg_problem_length += len(instance["problem_statement"])

    avg_problem_length = avg_problem_length / len(instances) if instances else 0

    return {
        "total_instances": len(instances),
        "unique_repos": len(repos),
        "repos": sorted(list(repos)),
        "avg_problem_length": int(avg_problem_length),
    }
