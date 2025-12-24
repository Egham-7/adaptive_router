# SWE-bench Benchmark (Modal Cloud)

Custom SWE-bench agent using Modal cloud execution with Nordlys intelligent routing.

## Quick Start

### 1. Install Dependencies
```bash
cd benchmarks
uv sync
```

### 2. Configure Modal
```bash
modal setup
```

### 3. Set Environment Variables
The `.env` file in `benchmarks/` should have:
```bash
NORDLYS_API_KEY=apk_...
NORDLYS_API_BASE=https://api.nordlyslabs.com
```

### 4. Run Benchmark
```bash
cd benchmarks

# Test (5 instances)
uv run python -m swe-bench-2.src.runner --slice 0:5

# Full run (500 instances)
uv run python -m swe-bench-2.src.runner
```

### 5. Submit for Evaluation
```bash
sb-cli submit swe-bench_verified test --predictions_path results/preds.json --run_id my-run
sb-cli get-report swe-bench_verified test --run_id my-run
```

---

## CLI Options

```bash
uv run python -m swe-bench-2.src.runner [OPTIONS]

Options:
  --subset     Dataset subset: verified, lite, full (default: verified)
  --split      Dataset split (default: test)
  --slice      Slice of instances, e.g., 0:10 (default: all)
  --model      LLM model (default: anthropic/nordlys/nordlys-code)
  --output     Output directory (default: results)
  --workers    Parallel workers (default: 4)
  --cost-limit Cost limit per instance (default: 3.0)
  --max-steps  Max agent steps (default: 50)
```

## Optimizations

The runner includes several optimizations for cost and speed:

| Feature | Description |
|---------|-------------|
| **Parallel execution** | 4 concurrent Modal sandboxes (configurable via `--workers`) |
| **Reduced idle timeout** | 90s instead of 600s (6.6x cost savings during LLM waits) |
| **Resource allocation** | 0.5 CPU, 512 MiB memory per sandbox |
| **Retry with backoff** | 3 retries with exponential backoff (1s, 2s, 4s) |
| **Crash recovery** | Resume from last saved prediction if interrupted |
| **Cost estimation** | Shows estimated Modal cost before running |
| **Budget warning** | Prompts if estimated cost exceeds $30 |

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   runner.py │────▶│   agent.py  │────▶│  modal_env  │
│             │     │  (LiteLLM)  │     │  (swe-rex)  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
  Load Dataset      LLM API Calls      Modal Cloud Exec
```

- **runner.py**: Orchestrates the benchmark run
- **agent.py**: Agent logic using LiteLLM for LLM calls
- **modal_env.py**: Modal deployment via swe-rex

---

## Output

```
results/
├── preds.json          # Predictions for sb-cli submission
└── trajs/              # Per-instance trajectories
    ├── instance1.json
    └── instance2.json
```


---

## Leaderboard Submission

1. Fork https://github.com/swe-bench/experiments
2. Create folder: `evaluation/verified/YYYYMMDD_nordlys-router/`
3. Add: `predictions.json`, `metadata.yaml`
4. Create PR with run_id
