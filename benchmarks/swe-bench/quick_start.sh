#!/bin/bash
# Quick start script for SWE-bench benchmarking

set -e

echo "==================================================================="
echo "  SWE-bench + Adaptive AI - Quick Start"
echo "==================================================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo ""
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "✓  Created .env file"
    echo ""
    echo "📝 Please edit .env and add your ADAPTIVE_API_KEY"
    echo ""
    exit 1
fi

# Check for ADAPTIVE_API_KEY
if ! grep -q "^ADAPTIVE_API_KEY=.\+" .env; then
    echo ""
    echo "⚠️  ADAPTIVE_API_KEY not set in .env"
    echo "📝 Please edit .env and add your API key"
    echo ""
    exit 1
fi

echo ""
echo "Step 1: Installing dependencies with uv..."
uv sync

echo ""
echo "Step 2: Verifying installation..."
uv run python -c "import openai; import pandas; import tiktoken; print('✓ All dependencies installed')"

echo ""
echo "Step 3: Verifying sb-cli..."
if uv run sb-cli --help &> /dev/null; then
    echo "✓ sb-cli is installed"
else
    echo "⚠️  sb-cli not found, installing..."
    uv pip install sb-cli
fi

echo ""
echo "Step 4: Downloading SWE-bench Lite dataset..."
if [ -f "data/swe-bench_lite.json" ]; then
    echo "✓ Dataset already downloaded"
else
    echo "Downloading dataset (this may take a few minutes)..."
    uv run python download_dataset.py
fi

echo ""
echo "==================================================================="
echo "  Setup Complete!"
echo "==================================================================="
echo ""
echo "Run a quick test:"
echo "  uv run python run_benchmark.py --quick"
echo ""
echo "See README.md for more options"
echo ""
