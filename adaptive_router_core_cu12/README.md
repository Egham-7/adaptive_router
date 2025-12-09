# Adaptive Router Core (CUDA 12.x)

GPU-accelerated C++ inference core for Adaptive Router with CUDA 12.x support.

## Overview

CUDA-enabled variant providing **40x speedup** for batch operations on NVIDIA GPUs. Automatically falls back to CPU when GPU unavailable. Same API as CPU variant.

## Requirements

- **Linux x86_64 only** (CUDA limitation)
- **CUDA 12.x compatible GPU** (optional - falls back to CPU)
- **Python 3.11+**

## Installation

```bash
pip install adaptive-router-core-cu12
```

## Usage

API identical to CPU variant:

```python
from adaptive_core_ext import Router
import numpy as np

router = Router.from_file("profile.json")

# Single routing (GPU-accelerated)
embedding = np.random.randn(384).astype(np.float32)
response = router.route(embedding, cost_bias=0.5)

# Batch routing (40x faster on GPU)
embeddings = np.random.randn(1000, 384).astype(np.float32)
responses = router.route_batch(embeddings, cost_bias=0.5)
```

## Performance (NVIDIA L40S)

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| Single route | 0.2ms | 0.1ms | 2x |
| Batch (1000) | 200ms | 5ms | **40x** |

## When to Use

- **Use this package**: Linux + NVIDIA GPU + high-throughput batch processing
- **Use CPU variant**: macOS/Windows or lower throughput requirements

## API Reference

See [adaptive-router-core README](adaptive_router_core/README.md) - API is identical.

## License

MIT License - see [LICENSE](../LICENSE)
