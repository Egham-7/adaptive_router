# SWE-bench with Nordlys

## Setup

1. Install:
   ```bash
   cd benchmarks
   uv sync
   uv tool install mini-swe-agent
   ```

2. Configure API:
   ```bash
   mini-extra config set ANTHROPIC_API_KEY "your-nordlys-api-key" #Anthropic is supported internally by mini-agent
   mini-extra config set ANTHROPIC_API_BASE "https://api.llmadaptive.uk"
   ```

## Run

```bash
# Test (1 instance)
uv run python swe-bench/swe-bench/src/run.py --skip-pricing --split test --slice 0:1 --output swe-bench/swe-bench/results/test

# Full run
uv run python swe-bench/swe-bench/src/run.py --skip-pricing --output swe-bench/swe-bench/results/full-run
```

## Options

| Option | Description |
|--------|-------------|
| `--skip-pricing` | Skip cost calculation |
| `--split` | test or dev |
| `--slice` | Instance range (e.g., `0:5`) |
| `--output` | Output directory |
| `--workers` | Parallel workers (default: 4) |

## Submit for Evaluation

After a run completes, submit results for evaluation:

```bash
# Submit results
uv run sb-cli submit swe-bench_verified test \
    --predictions_path results/my-run/all_preds.jsonl \
    --run_id my-run-name

# Check status
uv run sb-cli list-runs swe-bench_verified test

# Get results (after evaluation completes)
uv run sb-cli get-report swe-bench_verified test my-run-name
```


First time setup requires API key:
```bash
uv run sb-cli gen-api-key your.email@example.com
uv run sb-cli verify-api-key YOUR_CODE
```
