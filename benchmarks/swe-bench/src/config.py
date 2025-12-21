"""
Configuration settings for SWE-bench benchmarking.

This module provides centralized configuration for benchmark runs,
including model settings, API keys, and execution parameters.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv(".env.local")  # Override with local settings


@dataclass
class SWEBenchConfig:
    """Configuration for SWE-bench benchmark runs."""

    # SWE-bench API
    api_key: str | None = None

    # Dataset selection (use "verified" for leaderboard eligibility)
    dataset: str = "verified"  # Options: "verified", "lite", "full"
    split: str = "test"  # Verified uses test split
    max_instances: int = 500  # Number of instances (-1 for all, Verified has 500)

    # Generation settings
    temperature: float = 0.2  # Lower temp for more deterministic patches
    max_tokens: int = 4096  # Patches can be longer than code completions

    # Execution settings
    timeout_seconds: int = 300  # 5 minutes per test execution

    # Results storage
    results_folder: str = "results"

    # Retry configuration
    retry_max_attempts: int = 3
    retry_initial_seconds: float = 2.0
    retry_exp_base: float = 2.0
    retry_cap_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "SWEBenchConfig":
        """Create config from environment variables."""
        return cls(
            api_key=os.getenv("SWEBENCH_API_KEY"),
            dataset=os.getenv("SWEBENCH_DATASET", "verified"),
            split=os.getenv("SWEBENCH_SPLIT", "test"),
            max_instances=int(os.getenv("MAX_INSTANCES", "500")),
            temperature=float(os.getenv("GENERATION_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            timeout_seconds=int(os.getenv("DOCKER_TIMEOUT", "300")),
            results_folder=os.getenv("RESULTS_FOLDER", "results"),
            retry_max_attempts=int(os.getenv("RETRY_MAX_ATTEMPTS", "3")),
            retry_initial_seconds=float(os.getenv("RETRY_INITIAL_SECONDS", "2.0")),
            retry_exp_base=float(os.getenv("RETRY_EXP_BASE", "2.0")),
            retry_cap_seconds=float(os.getenv("RETRY_CAP_SECONDS", "30.0")),
        )

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError(
                "SWEBENCH_API_KEY not found in environment. "
                "Please set it in .env or pass it explicitly."
            )
        return True

    def get_results_path(self, model_name: str) -> Path:
        """Get results directory path for a specific model."""
        safe_name = model_name.replace(":", "_").replace("/", "_")
        return Path(self.results_folder) / safe_name


# Default models for Adaptive routing (Claude 4.5 only)
DEFAULT_MODELS = ["anthropic/claude-opus-4-5", "anthropic/claude-sonnet-4-5"]


@dataclass
class AdaptiveConfig:
    """Configuration for Adaptive routing model."""

    api_key: str | None = None
    api_base: str = "https://api.llmadaptive.uk"  # No /v1 suffix - Anthropic client adds it
    models: list[str] | None = None  # Defaults to DEFAULT_MODELS if None
    cost_bias: float = 0.5
    max_concurrent: int = 10
    max_requests_per_second: float = 20.0

    @classmethod
    def from_env(cls) -> "AdaptiveConfig":
        """Create config from environment variables."""
        # Parse models list from env, default to Claude 4.5 models
        models_str = os.getenv("ADAPTIVE_MODELS")
        models: list[str] | None = None
        if models_str:
            # Try to parse as JSON array first
            try:
                import json

                models = json.loads(models_str)
            except Exception:
                # Fall back to comma-separated
                models = [m.strip() for m in models_str.split(",")]
        else:
            # Use default Claude 4.5 models
            models = DEFAULT_MODELS.copy()

        return cls(
            api_key=os.getenv("ADAPTIVE_API_KEY"),
            api_base=os.getenv("ADAPTIVE_BASE_URL", "https://api.llmadaptive.uk"),
            models=models,
            cost_bias=float(os.getenv("ADAPTIVE_COST_BIAS", "0.5")),
            max_concurrent=int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),
            max_requests_per_second=float(os.getenv("MAX_REQUESTS_PER_SECOND", "20.0")),
        )

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError(
                "ADAPTIVE_API_KEY not found in environment. "
                "Please set it in .env or pass it explicitly."
            )
        return True


class BenchmarkSettings:
    """
    Complete benchmark settings manager.

    This class provides a unified interface to all configuration settings.
    """

    def __init__(self) -> None:
        """Initialize settings from environment."""
        self.swebench = SWEBenchConfig.from_env()
        self.adaptive = AdaptiveConfig.from_env()

    def validate(self) -> bool:
        """Validate all configurations."""
        self.swebench.validate()
        return self.adaptive.validate()

    def print_summary(self) -> None:
        """Print configuration summary to console."""
        # Dataset instance counts
        dataset_sizes = {"verified": 500, "lite": 300, "full": 2294}
        total = dataset_sizes.get(self.swebench.dataset, "unknown")

        print("\n" + "=" * 70)
        print("  SWE-bench Benchmark Configuration")
        print("=" * 70)
        print("\n  Benchmark Settings:")
        print(f"    Dataset:                  {self.swebench.dataset}")
        print(f"    Split:                    {self.swebench.split}")
        print(f"    Max instances:            {self.swebench.max_instances} (of {total} total)")
        print(f"    Temperature:              {self.swebench.temperature}")
        print(f"    Max tokens:               {self.swebench.max_tokens}")
        print("    Evaluation:               Cloud-based (sb-cli)")
        print(f"    Results folder:           {self.swebench.results_folder}")

        print("\n  Adaptive Router:")
        if self.adaptive.models:
            print(f"    Models:                   {len(self.adaptive.models)} models")
            print(f"      - Cost bias:            {self.adaptive.cost_bias}")
            print(f"      - Models:               {', '.join(self.adaptive.models)}")
        else:
            print("    Models:                   Auto (server default)")
            print(f"      - Cost bias:            {self.adaptive.cost_bias}")

        print("\n" + "=" * 70 + "\n")


# Global settings instance
settings = BenchmarkSettings()


# Convenience functions
def get_swebench_config() -> SWEBenchConfig:
    """Get SWE-bench configuration."""
    return settings.swebench


def get_adaptive_config() -> AdaptiveConfig:
    """Get Adaptive configuration."""
    return settings.adaptive
