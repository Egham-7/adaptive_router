"""Adaptive Router Benchmarking Suite.

This package provides benchmarking tools for evaluating adaptive router
performance on on-device AI workloads.
"""

__version__ = "0.1.0"

from adaptive_router_benchmarks.core.routers import CactusProfileRouter, ClaudeOracleRouter
from adaptive_router_benchmarks.core.runner import BenchmarkRunner
from adaptive_router_benchmarks.core.metrics import BenchmarkMetrics

__all__ = [
    "CactusProfileRouter",
    "ClaudeOracleRouter",
    "BenchmarkRunner",
    "BenchmarkMetrics",
]
