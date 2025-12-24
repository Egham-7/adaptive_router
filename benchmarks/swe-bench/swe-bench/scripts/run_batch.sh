#!/bin/bash
# Run full SWE-bench Verified benchmark (500 instances)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$BENCHMARKS_DIR"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

echo "Starting SWE-bench Verified benchmark (500 instances)..."
echo "Model: anthropic/nordlys/nordlys-code"
echo "Deployment: Modal cloud"

uv run python -m swe-bench-2.src.runner \
    --output swe-bench-2/results

echo "Benchmark complete! Submit results with:"
echo "  sb-cli submit swe-bench_verified test \\"
echo "    --predictions_path swe-bench-2/results/preds.json \\"
echo "    --run_id my-run"
