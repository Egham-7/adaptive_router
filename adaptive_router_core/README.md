# Adaptive Router Core

High-performance C++ inference core for Adaptive Router with Python bindings and C FFI API.

## Overview

C++20 implementation providing **10x performance** improvements over pure Python. Features zero-copy Python integration, cross-platform support, and zero heap allocations per request.

## Requirements

- **C++20 compiler**: GCC 10+, Clang 12+, or MSVC 2019 16.11+
- **CMake** 3.24+
- **Conan** 2.x (package manager)
- **CUDA** 11.8+ (optional, for GPU acceleration)

## Installation

### Python Package (Recommended)
```bash
pip install adaptive-router-core
```

### Build from Source

**Linux/macOS:**
```bash
cd adaptive_router_core
conan install . --output-folder=build --build=missing --settings=build_type=Release
cmake --preset conan-release
cmake --build build/build/Release
```

**Windows:**
```bash
cd adaptive_router_core
conan install . --output-folder=build --build=missing --settings=build_type=Release
cmake --preset conan-default
cmake --build build
```

### CMake Options
- `ADAPTIVE_ENABLE_CUDA=AUTO/ON/OFF` - Enable CUDA GPU acceleration
- `ADAPTIVE_BUILD_PYTHON=ON/OFF` - Build Python bindings
- `ADAPTIVE_BUILD_TESTS=ON/OFF` - Build test suite

## Usage

### Python (nanobind)
```python
from adaptive_core_ext import Router
import numpy as np

router = Router.from_file("profile.json")
embedding = np.random.randn(384).astype(np.float32)
response = router.route(embedding, cost_bias=0.5)

print(f"Selected: {response.selected_model}")
print(f"Cluster: {response.cluster_id}")
```

### C++ API
```cpp
#include "router.hpp"

auto router = Router::from_file("profile.json");
std::vector<float> embedding(384);
RouteRequest request{.embedding = embedding, .cost_bias = 0.5f};
auto response = router.route(request);

std::cout << "Selected: " << response.selected_model << std::endl;
```

### C FFI API
```c
#include "adaptive.h"

AdaptiveRouter* router = adaptive_router_create("profile.json");
float embedding[384] = {...};
AdaptiveRouteResult* result = adaptive_router_route(router, embedding, 384, 0.5f);

printf("Selected: %s\n", result->selected_model);
adaptive_route_result_free(result);
adaptive_router_destroy(router);
```

## Performance

| Implementation | Latency | Throughput | Speedup |
|----------------|---------|------------|---------|
| Pure Python | ~2ms | 500 req/s | 1x |
| C++ Core | ~0.2ms | 5,000 req/s | **10x** |

## API Reference

### Router Class
- `Router.from_file(path)` - Load from JSON profile
- `Router.from_binary(path)` - Load from MessagePack profile
- `route(embedding, cost_bias)` - Route single embedding
- `route_batch(embeddings, cost_bias)` - Route multiple embeddings

### RouteResponse
- `selected_model: str` - Selected model ID
- `alternatives: list[str]` - Alternative models
- `cluster_id: int` - Assigned cluster
- `cluster_distance: float` - Distance to cluster centroid

## License

MIT License - see [LICENSE](../LICENSE)
