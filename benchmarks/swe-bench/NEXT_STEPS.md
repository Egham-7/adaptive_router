# Next Steps - SWE-bench Implementation

## Phase 1: Setup & Testing (DO THIS NOW)

### 1. Run the setup script
```bash
cd swe-bench
./quick_start.sh
```

This will:
- Check for `.env` file (create from template if missing)
- Install dependencies with `uv sync`
- Verify SWE-bench CLI is installed

### 2. Configure your API key

Edit `.env` and add your Adaptive AI API key:
```bash
ADAPTIVE_API_KEY=your_actual_key_here
```

### 3. Test the installation

```bash
# Verify everything works
uv run mypy --version
uv run python -c "import openai; print('✓ OpenAI client works')"

# Check SWE-bench CLI
swebench --help
```

### 4. Run a quick test

```bash
uv run python run_benchmark.py --quick
```

**Expected behavior:**
- Should initialize Adaptive model
- Generate patches for 5 test instances
- Print results summary with costs and model selections
- Save results to `results/adaptive/`

**Note:** The current implementation uses mock data. See Phase 2 for real SWE-bench integration.

---

## Phase 2: Real SWE-bench Integration (AFTER TESTING)

The current implementation works end-to-end but uses placeholder data. To integrate with real SWE-bench:

### 1. Download SWE-bench dataset

```bash
# Using SWE-bench CLI
swebench download --dataset lite --output-dir ./data
```

### 2. Update `load_swebench_dataset()` in `run_benchmark.py`

Replace the mock data loader (line ~50) with:

```python
def load_swebench_dataset(dataset: str, max_instances: int) -> list[dict]:
    """Load real SWE-bench dataset."""
    from swebench import get_dataset

    # Load dataset
    instances = get_dataset(dataset)

    # Limit if requested
    if max_instances > 0:
        instances = instances[:max_instances]

    return instances
```

### 3. Add patch application & test execution

Update `run_benchmark()` function (line ~90) to use SWE-bench CLI:

```python
# After generating patch
if metrics.patch_generated:
    # Apply patch using SWE-bench
    from swebench import apply_patch, run_tests

    patch_path = save_patch_to_file(metrics.patch_content, instance_id)

    # Apply and test
    apply_result = apply_patch(patch_path, instance)
    if apply_result.success:
        test_result = run_tests(instance)
        metrics.test_passed = test_result.passed
        metrics.test_output = test_result.output
        metrics.resolution_status = "resolved" if test_result.passed else "failed"
    else:
        metrics.resolution_status = "failed"
        metrics.error_message = apply_result.error
```

### 4. Add Docker support

Make sure Docker is installed and running for test isolation:

```bash
docker --version
```

Update `config.py` to use Docker settings from environment.

---

## Phase 3: Optimization & Analysis

### 1. Cost Analysis

After running benchmarks, analyze which models work best:

```python
# Add to run_benchmark.py
def analyze_model_performance(tracker: ResultTracker):
    """Analyze which models are most cost-effective."""
    results = tracker.instance_results

    by_model = {}
    for r in results:
        model = r.generation_metrics.model_used
        if model not in by_model:
            by_model[model] = {"resolved": 0, "total": 0, "cost": 0}

        by_model[model]["total"] += 1
        by_model[model]["cost"] += r.total_cost
        if r.resolution_status == "resolved":
            by_model[model]["resolved"] += 1

    # Print analysis
    for model, stats in by_model.items():
        resolution_rate = stats["resolved"] / stats["total"] * 100
        avg_cost = stats["cost"] / stats["total"]
        print(f"{model}: {resolution_rate:.1f}% resolved, ${avg_cost:.4f} per instance")
```

### 2. Multi-step Reasoning (Optional)

For better results, implement multi-step approach:

1. **Analyze issue** (cheap model)
2. **Explore codebase** (use search/grep)
3. **Plan fix** (medium model)
4. **Generate patch** (appropriate model based on complexity)
5. **Debug if needed** (only if tests fail)

This would be a new `MultiStepAdaptiveModel` class.

### 3. Repository-specific tuning

Different repos may need different approaches:
- Django: Needs understanding of ORM patterns
- NumPy: Requires C extension knowledge
- Requests: HTTP/networking expertise

Add repo detection and specialized prompts.

---

## Phase 4: Production Features

### 1. Caching

Add caching to avoid re-generating patches:

```python
import hashlib
import json

def get_cache_key(problem_statement: str, repo: str) -> str:
    content = f"{repo}:{problem_statement}"
    return hashlib.md5(content.encode()).hexdigest()

# Before generating, check cache
cache_dir = Path("cache")
cache_file = cache_dir / f"{cache_key}.json"
if cache_file.exists():
    return load_cached_result(cache_file)
```

### 2. Parallel execution

Process multiple instances in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(solve_instance, inst) for inst in instances]
    results = [f.result() for f in futures]
```

### 3. Progress tracking

Add progress bar:

```bash
pip install tqdm
```

```python
from tqdm import tqdm

for instance in tqdm(instances, desc="Processing"):
    # ... solve instance
```

### 4. Retry logic

Add retries for API failures:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def generate_patch_with_retry(self, ...):
    return self.generate_patch(...)
```

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution:** Run from the `swe-bench/` directory:
```bash
cd swe-bench
uv run python run_benchmark.py --quick
```

### Issue: "ADAPTIVE_API_KEY not found"

**Solution:** Create `.env` file:
```bash
cp .env.example .env
# Edit .env and add your key
```

### Issue: Type errors with mypy

**Solution:** Add type annotations or use type: ignore:
```python
# Add proper types
def foo(x: int) -> str:
    return str(x)

# Or ignore specific lines
result = some_function()  # type: ignore
```

### Issue: SWE-bench CLI not found

**Solution:** Install globally or in environment:
```bash
pip install swebench
# or
uv pip install swebench
```

---

## Testing Checklist

- [ ] Setup script runs without errors
- [ ] `.env` file created with API key
- [ ] `uv sync` installs all dependencies
- [ ] `mypy src/` passes type checking
- [ ] `--quick` mode runs and generates results
- [ ] Results saved to JSON and CSV
- [ ] Console output shows model selection stats
- [ ] Cost tracking appears accurate

---

## Resources

- **SWE-bench Docs**: https://www.swebench.com/
- **SWE-bench CLI**: https://www.swebench.com/sb-cli/installation/
- **SWE-bench Paper**: https://arxiv.org/abs/2310.06770
- **Adaptive AI**: https://adaptive.ai/

---

## Questions?

If you encounter issues:
1. Check the logs in your console output
2. Verify `.env` configuration
3. Try `--quick` mode first before full benchmarks
4. Check that Docker is running (for real SWE-bench tests)

Good luck! 🚀
