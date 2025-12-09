# Adaptive Router

Intelligent LLM model selection via ML-powered clustering. Reduce costs by 30-70% using cluster-based routing with per-cluster error rates and prompt analysis.

[![PyPI](https://img.shields.io/pypi/v/adaptive-router)](https://pypi.org/project/adaptive-router/)
[![Python](https://img.shields.io/pypi/pyversions/adaptive-router)](https://pypi.org/project/adaptive-router/)
[![License](https://img.shields.io/github/license/Egham-7/adaptive)](https://github.com/Egham-7/adaptive/blob/main/LICENSE)

## What It Does

1. **Semantic Clustering**: Groups similar prompts using sentence embeddings
2. **Performance Tracking**: Maintains per-cluster error rates for each model
3. **Smart Selection**: Balances quality vs. cost using configurable `cost_bias`

## Installation

**Note**: Adaptive Router requires either the CPU or CUDA C++ core to function.

### CPU Version (Recommended)
```bash
pip install adaptive-router[cpu]
```

### CUDA Version (GPU Accelerated)
```bash
pip install adaptive-router[cu12]
```

### From Source (Development)
```bash
git clone https://github.com/Egham-7/adaptive
cd adaptive
uv sync --package adaptive-router  # Includes CPU core
```

**Requirements**: Python 3.11+, CMake (for compilation), optional: CUDA 12.x (for GPU version)

## Quick Start

### Basic Usage
```python
from adaptive_router import ModelRouter

# Load trained profile
router = ModelRouter.from_json_file("profile.json")

# Select optimal model
model_id = router.route("Write a Python sorting function", cost_bias=0.3)
print(model_id)  # "openai/gpt-3.5-turbo"
```

### Advanced Usage
```python
from adaptive_router import ModelSelectionRequest

response = router.select_model(
    ModelSelectionRequest(
        prompt="Design a distributed system",
        cost_bias=0.9  # Prefer quality over cost
    )
)

print(f"Selected: {response.model_id}")
print(f"Alternatives: {[alt.model_id for alt in response.alternatives]}")
```

### Training
```bash
# Train from CSV dataset
uv run python adaptive_router/train/train.py \
  --config adaptive_router/train/examples/configs/train_minimal.toml
```

## Key Features

- **30-70% Cost Reduction** vs. always using GPT-4
- **10x Performance** with C++ inference core
- **Multi-Provider Support**: OpenAI, Anthropic, and custom models
- **Cloud Storage**: MinIO/S3 profile storage
- **Production Ready**: Docker, Railway, and bare-metal deployment

## API Overview

### Core Classes
- `ModelRouter`: Main routing interface
- `ModelSelectionRequest`: Routing request with prompt and preferences
- `ModelSelectionResponse`: Routing result with selected model and alternatives

### Factory Methods
- `ModelRouter.from_json_file(path)`: Load from JSON
- `ModelRouter.from_minio(settings)`: Load from MinIO/S3
- `ModelRouter.from_profile(profile)`: Load from memory

## Links

- 📖 **[Full Documentation](https://docs.llmadaptive.uk)**
- 🚀 **[Training Guide](adaptive_router/train/examples/README.md)**
- 🏗️ **[Architecture](ARCHITECTURE.md)**
- 🐛 **[Issues](https://github.com/Egham-7/adaptive/issues)**
- 💬 **[Discussions](https://github.com/Egham-7/adaptive/discussions)**

## License

MIT License - see [LICENSE](LICENSE)
