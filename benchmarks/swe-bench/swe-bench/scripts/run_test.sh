#!/bin/bash
# Test run with N SWE-bench instances (default: 5)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$BENCHMARKS_DIR"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

SLICE="${1:-5}"

echo "Starting SWE-bench test run ($SLICE instances)..."
echo "Model: anthropic/nordlys/nordlys-code"
echo "Deployment: Modal cloud"

uv run python -m swe-bench-2.src.runner \
    --slice "0:$SLICE" \
    --output swe-bench-2/results

echo "Test complete! Check swe-bench-2/results/preds.json for results."
