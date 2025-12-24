#!/usr/bin/env python3
"""SWE-bench runner using Modal cloud execution."""

import argparse
import asyncio
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from dotenv import load_dotenv

from .agent import AgentConfig, SWEAgent
from .modal_env import ModalConfig, ModalEnvironment, get_docker_image, retry_with_backoff


class PredictionSaver:
    """Thread-safe incremental prediction saver for crash recovery."""

    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.predictions: list[dict] = []
        self._lock = threading.Lock()

        # Load existing predictions if resuming
        if output_file.exists():
            try:
                with open(output_file) as f:
                    self.predictions = json.load(f)
                print(f"Resuming: loaded {len(self.predictions)} existing predictions")
            except json.JSONDecodeError:
                self.predictions = []

    def add(self, prediction: dict) -> None:
        """Add prediction and save atomically."""
        with self._lock:
            self.predictions.append(prediction)
            self._save()

    def get_completed_ids(self) -> set[str]:
        """Get IDs of already completed instances."""
        return {p["instance_id"] for p in self.predictions}

    def _save(self) -> None:
        """Save predictions atomically using temp file."""
        temp_file = self.output_file.with_suffix(".json.tmp")
        with open(temp_file, "w") as f:
            json.dump(self.predictions, f, indent=2)
        temp_file.rename(self.output_file)  # Atomic on POSIX


def load_instances(subset: str = "verified", split: str = "test", slice_str: str = None):
    """Load SWE-bench instances from HuggingFace."""
    dataset_name = f"princeton-nlp/SWE-bench_{subset.capitalize()}"
    print(f"Loading dataset: {dataset_name}")

    dataset = load_dataset(dataset_name, split=split)

    if slice_str:
        # Parse slice like "0:10" or ":5"
        parts = slice_str.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(dataset)
        dataset = dataset.select(range(start, min(end, len(dataset))))

    print(f"Loaded {len(dataset)} instances")
    return dataset


async def solve_instance(
    instance: dict,
    agent: SWEAgent,
    output_dir: Path,
) -> dict:
    """Solve a single SWE-bench instance.

    Returns:
        Prediction dict with instance_id, model_patch, etc.
    """
    instance_id = instance["instance_id"]
    print(f"\n{'='*60}")
    print(f"Solving: {instance_id}")
    print(f"{'='*60}")

    # Get Docker image for this instance
    image = get_docker_image(instance_id)

    # Create Modal environment with instance-specific image
    config = ModalConfig(
        image=image,
        startup_timeout=300,  # 5 min for image pull
        runtime_timeout=300,  # 5 min per request
        deployment_timeout=3600,  # 1 hour max
    )

    patch = ""
    cost = 0.0

    async def run_with_env():
        """Run agent in Modal environment (for retry wrapper)."""
        async with ModalEnvironment(config) as env:
            return await agent.solve(
                problem_statement=instance["problem_statement"],
                run_command=env.run_command,
                get_patch=env.get_patch,
            )

    try:
        patch, cost = await retry_with_backoff(
            run_with_env,
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay,
        )
    except Exception as e:
        print(f"Error solving {instance_id} after retries: {e}")
        patch = ""
        cost = 0.0

    # Create prediction
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": agent.config.model,
        "model_patch": patch,
    }

    # Save trajectory
    traj_dir = output_dir / "trajs"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_file = traj_dir / f"{instance_id}.json"

    trajectory = {
        "instance_id": instance_id,
        "model": agent.config.model,
        "patch": patch,
        "cost": cost,
        "timestamp": datetime.now().isoformat(),
    }
    with open(traj_file, "w") as f:
        json.dump(trajectory, f, indent=2)

    print(f"Completed: {instance_id} (cost: ${cost:.4f})")
    return prediction


async def run_batch(
    instances,
    agent: SWEAgent,
    output_dir: Path,
    workers: int = 4,
) -> list[dict]:
    """Run batch of instances with parallel workers.

    Uses asyncio.Semaphore to limit concurrent Modal sandboxes,
    preventing cost overruns and rate limiting issues.
    """
    # Thread-safe saver with crash recovery
    preds_file = output_dir / "preds.json"
    saver = PredictionSaver(preds_file)

    # Skip already completed instances (for resume)
    completed = saver.get_completed_ids()
    remaining = [i for i in instances if i["instance_id"] not in completed]

    if len(completed) > 0:
        print(f"Skipping {len(completed)} already completed instances")

    if not remaining:
        print("All instances already completed!")
        return saver.predictions

    print(f"Processing {len(remaining)} instances with {workers} workers")

    # Semaphore limits concurrent sandboxes
    semaphore = asyncio.Semaphore(workers)

    async def process_instance(instance: dict) -> Optional[dict]:
        """Process single instance with semaphore-controlled concurrency."""
        async with semaphore:
            try:
                pred = await solve_instance(instance, agent, output_dir)
                saver.add(pred)
                return pred
            except Exception as e:
                print(f"Failed {instance['instance_id']}: {e}")
                # Still save a failed prediction for tracking
                failed_pred = {
                    "instance_id": instance["instance_id"],
                    "model_name_or_path": agent.config.model,
                    "model_patch": "",
                    "error": str(e),
                }
                saver.add(failed_pred)
                return failed_pred

    # Launch all tasks (semaphore controls actual concurrency)
    tasks = [process_instance(instance) for instance in remaining]

    # Wait for all with progress reporting
    completed_count = 0
    total = len(tasks)

    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed_count += 1
        if result:
            print(f"Progress: {completed_count}/{total} instances completed")

    return saver.predictions


def estimate_modal_cost(
    num_instances: int,
    workers: int,
    avg_seconds_per_instance: float = 120.0,
    cpu_cores: float = 0.5,
    memory_mb: int = 512,
) -> dict:
    """Estimate Modal compute cost for a batch run.

    Modal pricing (approximate as of 2024):
    - CPU: $0.000463/core/minute (~$0.028/core/hour)
    - Memory: $0.000031/MB/minute (~$0.0019/MB/hour)
    - Minimum charge: ~$0.001 per sandbox start

    Returns:
        Dict with estimated costs breakdown
    """
    total_compute_seconds = num_instances * avg_seconds_per_instance

    # Cost per compute-second
    cpu_cost_per_sec = 0.000463 * cpu_cores / 60
    mem_cost_per_sec = 0.000031 * memory_mb / 60

    cpu_cost = total_compute_seconds * cpu_cost_per_sec
    mem_cost = total_compute_seconds * mem_cost_per_sec
    startup_cost = num_instances * 0.001

    total = cpu_cost + mem_cost + startup_cost

    # Wall clock estimate
    parallel_factor = min(workers, num_instances)
    wall_clock_minutes = (total_compute_seconds / parallel_factor) / 60

    return {
        "num_instances": num_instances,
        "workers": workers,
        "estimated_cpu_cost": round(cpu_cost, 4),
        "estimated_memory_cost": round(mem_cost, 4),
        "estimated_startup_cost": round(startup_cost, 4),
        "estimated_total_cost": round(total, 4),
        "estimated_wall_clock_minutes": round(wall_clock_minutes, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="SWE-bench runner with Modal")
    parser.add_argument(
        "--subset",
        default="verified",
        choices=["verified", "lite", "full"],
        help="SWE-bench subset",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--slice", default=None, help="Slice of instances (e.g., 0:10)")
    parser.add_argument(
        "--model",
        default="anthropic/nordlys/nordlys-code",
        help="Model to use",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=3.0,
        help="Cost limit per instance",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max steps per instance",
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load instances
    instances = load_instances(args.subset, args.split, args.slice)

    # Create agent (use NORDLYS_ env vars)
    agent_config = AgentConfig(
        model=args.model,
        api_base=os.environ.get("NORDLYS_API_BASE"),
        api_key=os.environ.get("NORDLYS_API_KEY"),
        max_steps=args.max_steps,
        cost_limit=args.cost_limit,
    )
    agent = SWEAgent(agent_config)

    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Subset: {args.subset}")
    print(f"  Split: {args.split}")
    print(f"  Slice: {args.slice or 'all'}")
    print(f"  Instances: {len(instances)}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {output_dir}")
    print(f"  Cost limit: ${args.cost_limit}")
    print(f"  Max steps: {args.max_steps}")

    # Estimate and display costs before running
    estimate = estimate_modal_cost(
        num_instances=len(instances),
        workers=args.workers,
    )
    print(f"\n--- Modal Cost Estimate ---")
    print(f"  CPU cost: ${estimate['estimated_cpu_cost']:.4f}")
    print(f"  Memory cost: ${estimate['estimated_memory_cost']:.4f}")
    print(f"  Startup cost: ${estimate['estimated_startup_cost']:.4f}")
    print(f"  ESTIMATED TOTAL: ${estimate['estimated_total_cost']:.2f}")
    print(f"  Estimated time: ~{estimate['estimated_wall_clock_minutes']:.0f} minutes")
    print(f"----------------------------")

    # Budget check
    BUDGET_LIMIT = 30.0  # Modal free tier
    if estimate["estimated_total_cost"] > BUDGET_LIMIT:
        print(f"\nWARNING: Estimated cost ${estimate['estimated_total_cost']:.2f} exceeds budget ${BUDGET_LIMIT}")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return

    print()

    # Run
    predictions = asyncio.run(
        run_batch(instances, agent, output_dir, args.workers)
    )

    print(f"\n{'='*60}")
    print(f"Completed {len(predictions)} instances")
    print(f"Predictions saved to: {output_dir / 'preds.json'}")
    print(f"\nTo submit:")
    print(f"  sb-cli submit swe-bench_{args.subset} {args.split} \\")
    print(f"    --predictions_path {output_dir / 'preds.json'} \\")
    print(f"    --run_id my-run")


if __name__ == "__main__":
    main()
