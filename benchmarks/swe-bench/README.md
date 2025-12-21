# SWE-bench Verified Benchmarking for Adaptive AI

Benchmark the Adaptive AI router on **SWE-bench Verified** (500 human-verified GitHub issues) with cloud-based evaluation via `sb-cli`. Results are eligible for the official SWE-bench leaderboard.

## Why SWE-bench Verified?

- **500 verified instances** - Human-confirmed solvable problems
- **Leaderboard eligible** - Submit results to official SWE-bench leaderboard
- **Real GitHub issues** - From popular Python repositories
- **Cloud evaluation** - No Docker setup needed
- **Cost tracking** - See exactly which models Adaptive selects

## Quick Start

### 1. Install Dependencies

```bash
cd swe-bench
uv sync
```

This installs:
- `sb-cli` - SWE-bench cloud evaluation tool
- `datasets` - HuggingFace datasets library
- `openai` - For Adaptive API
- All other dependencies

### 2. Configure API Keys

Create a `.env` file:
```bash
# Adaptive AI
ADAPTIVE_API_KEY=your_adaptive_api_key
ADAPTIVE_BASE_URL=https://api.llmadaptive.uk/v1

# SWE-bench (get from https://www.swebench.com/)
SWEBENCH_API_KEY=your_swebench_api_key
```

### 3. Run Benchmark

```bash
# Quick test (5 instances)
uv run python run_benchmark.py --quick

# Or run full leaderboard benchmark
uv run python run_benchmark.py --full
```

## Usage

### Quick Test (5 instances)

```bash
uv run python run_benchmark.py --quick
```

**Expected time:** ~2-5 minutes
**Expected cost:** ~$0.10-$0.30

### Medium Test (50 instances)

```bash
uv run python run_benchmark.py --medium
```

**Expected time:** ~30-60 minutes
**Expected cost:** ~$1.00-$3.00

### Full Benchmark (all 500 instances)

```bash
uv run python run_benchmark.py --full
```

**Expected time:** ~3-5 hours
**Expected cost:** ~$10-$30

### Advanced Options

```bash
# Custom number of instances
uv run python run_benchmark.py --max-instances 10

# Optimize for cost (use cheaper models more often)
uv run python run_benchmark.py --quick --cost-bias 0.8

# Optimize for quality (use better models more often)
uv run python run_benchmark.py --quick --cost-bias 0.2

# Skip submission (just generate predictions)
uv run python run_benchmark.py --quick --skip-submit

# Wait for evaluation results
uv run python run_benchmark.py --quick --wait
```

## How It Works

### 1. Patch Generation (Your Cost)

```
For each SWE-bench instance:
  ├─ Load problem statement from HuggingFace
  ├─ Adaptive routes to best Claude model
  │   ├─ Simple issues → claude-sonnet-4-5
  │   └─ Complex issues → claude-opus-4-5
  ├─ Generate patch
  └─ Track cost + tokens + model selection
```

### 2. Evaluation (Free - Cloud-based)

```
sb-cli submits predictions to SWE-bench:
  ├─ Clones repositories
  ├─ Applies your patches
  ├─ Runs test suites in Docker
  └─ Returns resolution results
```

## Output Files

After running, you'll get:

```
swe-bench/
├── predictions/
│   └── adaptive_20250115_143022.json    # Predictions for SWE-bench
├── results/adaptive/
│   ├── adaptive_20250115_143022_generation.json   # Detailed generation metrics
│   ├── adaptive_20250115_143022_generation.csv    # Summary CSV
│   └── adaptive_20250115_143022_report.json       # SWE-bench evaluation results
```

### Example Output

```
======================================================================
  SWE-bench Benchmark Configuration
======================================================================

  Benchmark Settings:
    Dataset:                  verified
    Split:                    test
    Max instances:            5 (of 500 total)
    Temperature:              0.2
    Max tokens:               4096
    Evaluation:               Cloud-based (sb-cli)

  Adaptive Router:
    Models:                   2 models
      - Cost bias:            0.5
      - Models:               claude-opus-4-5, claude-sonnet-4-5

======================================================================
Generating predictions for 5 instances
======================================================================

[1/5] Processing django__django-11099
  ✓ Patch generated | Cost: $0.0156 | Tokens: 1245 | Model: claude-sonnet-4-5

[2/5] Processing pytest-dev__pytest-5221
  ✓ Patch generated | Cost: $0.0423 | Tokens: 2134 | Model: claude-opus-4-5

...

======================================================================
  SWE-bench Benchmark Results
======================================================================

  Resolution Results:
    Total instances:      5
    Patches generated:    5

  Cost Metrics:
    Total cost:           $0.1487
    Cost per instance:    $0.0297

  Token Metrics:
    Total tokens:         15,234
    Input tokens:         10,123
    Output tokens:        5,111

  Model Selection (Adaptive):
    claude-sonnet-4-5                      3 ( 60.0%)
    claude-opus-4-5                        2 ( 40.0%)

======================================================================
```

## Cost Comparison

| Approach | Cost per Instance | Total (500 instances) |
|----------|------------------|----------------------|
| Always Claude Opus 4.5 | ~$0.045 | **~$22.50** |
| Always Claude Sonnet 4.5 | ~$0.015 | **~$7.50** |
| **Adaptive Router** | ~$0.025 | **~$12.50** |

**Adaptive achieves optimal quality while routing simpler tasks to cost-effective models!**

### Expected Model Selection (cost_bias=0.5)

- 60% → claude-sonnet-4-5 (simpler issues)
- 40% → claude-opus-4-5 (complex issues)

## Analyzing Results

### Check Evaluation Status

```bash
# List your runs
sb-cli list-runs swe-bench_verified test

# Get specific report
sb-cli get-report swe-bench_verified test adaptive_20250115_143022
```

### Load Results in Python

```python
import json
import pandas as pd

# Load generation metrics
with open("results/adaptive/adaptive_20250115_143022_generation.json") as f:
    data = json.load(f)

print(f"Total cost: ${data['cost_metrics']['total_cost_usd']}")
print(f"Resolution rate: {data['summary']['resolution_rate_percent']}%")

# Model selection stats
for model, stats in data['model_selection_stats']['models'].items():
    print(f"{model}: {stats['percentage']}%")

# Load CSV for analysis
df = pd.read_csv("results/adaptive/adaptive_20250115_143022_generation.csv")
print(df[['instance_id', 'cost_usd', 'model_used', 'resolution_status']])
```

## FAQ

### How is the dataset loaded?

The dataset is loaded automatically from HuggingFace using the `datasets` library:
```python
from datasets import load_dataset
dataset = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
```
HuggingFace handles caching automatically in `~/.cache/huggingface/`.

### How long does evaluation take?

- Cloud evaluation: 10-30 minutes after submission
- Use `--wait` flag to wait for results automatically

### Which models are used?

The benchmark uses only Claude 4.5 models via Adaptive routing:
- `claude-opus-4-5` - For complex issues
- `claude-sonnet-4-5` - For simpler issues

### How accurate is cost tracking?

Very accurate! We track:
- Actual tokens used from API responses
- Real costs based on current pricing
- Which specific model was selected for each request

### How do I submit to the leaderboard?

Results from `--full` runs on SWE-bench Verified are automatically eligible for the leaderboard when submitted via sb-cli.

## Troubleshooting

### "SWEBENCH_API_KEY not found"

Make sure `.env` exists and has your key:
```bash
cat .env | grep SWEBENCH_API_KEY
```

### "Failed to load dataset"

The HuggingFace datasets library will automatically download on first run. Ensure you have internet connectivity.

### Type errors with mypy

```bash
uv run mypy src/
```

## Development Workflow

### 1. Test Locally First

```bash
# Quick test to verify everything works
uv run python run_benchmark.py --quick --skip-submit
```

### 2. Iterate on Cost Bias

```bash
# Test different routing strategies
uv run python run_benchmark.py --max-instances 10 --cost-bias 0.2  # Quality
uv run python run_benchmark.py --max-instances 10 --cost-bias 0.5  # Balanced
uv run python run_benchmark.py --max-instances 10 --cost-bias 0.8  # Cost
```

### 3. Run Full Benchmark

```bash
# When you're confident in your config
uv run python run_benchmark.py --full
```

## Resources

- **SWE-bench**: https://www.swebench.com/
- **SWE-bench Leaderboard**: https://www.swebench.com/
- **SWE-bench Datasets Guide**: https://www.swebench.com/SWE-bench/guides/datasets/
- **sb-cli Docs**: https://www.swebench.com/sb-cli/
- **SWE-bench Paper**: https://arxiv.org/abs/2310.06770

## Next Steps

1. **Run quick test**: `uv run python run_benchmark.py --quick`
2. **Analyze results**: Check `results/adaptive/` directory
3. **Optimize routing**: Try different `--cost-bias` values
4. **Full benchmark**: Run with `--full` for leaderboard submission
