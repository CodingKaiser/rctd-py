# rctd-py Local Development

## GitHub CLI

`gh` is installed in the conda env `ps_pgueguen`. Always activate it before running `gh`:

```bash
eval "$(/usr/local/ngseq/miniforge3/bin/conda shell.bash hook)"
conda activate ps_pgueguen
```

## Architecture

GPU-accelerated Python reimplementation of spacexr RCTD for spatial transcriptomics deconvolution.

### Key source files

| File | Purpose |
|------|---------|
| `src/rctd/_irwls.py` | Core IRWLS solver — hot path, ~60% of runtime |
| `src/rctd/_likelihood.py` | Poisson-Lognormal likelihood with cubic spline interpolation (`calc_q_all`) |
| `src/rctd/_simplex.py` | Simplex projection for weight constraints |
| `src/rctd/_full.py` | Full-mode pipeline (all cell types per pixel) |
| `src/rctd/_doublet.py` | Doublet-mode pipeline (top 2 cell types per pixel) |
| `src/rctd/_multi.py` | Multi-mode pipeline (variable number of types) |
| `src/rctd/_sigma.py` | Sigma estimation (noise parameter) |
| `src/rctd/_rctd.py` | Top-level `run_rctd()` entry point |
| `src/rctd/cli.py` | Click CLI: `rctd info`, `rctd validate`, `rctd run` |
| `src/rctd/_types.py` | Config, result types, `auto_batch_size`, `resolve_device` |

### Solver pipeline

1. **Sigma estimation** (`_sigma.py`): Find optimal noise parameter
2. **Full mode** (`_full.py`): Unconstrained IRWLS → all K cell type weights per pixel
3. **Doublet/Multi mode** (`_doublet.py`, `_multi.py`): Select top types → constrained IRWLS

The IRWLS solver (`solve_irwls_batch_shared`) is the innermost hot loop:
- Iteratively solves weighted least squares with Poisson-Lognormal likelihood
- Each iteration: predict → derivatives → Hessian → PSD projection → box-QP → update
- Active pixel compaction skips converged pixels

## CLI

Entry point `rctd` is registered in `pyproject.toml` via `[project.scripts]`. Three subcommands:

- `rctd info` — environment info (versions, GPU detection), `--json` for machine-readable
- `rctd validate` — pre-flight checks on h5ad inputs (fast, no GPU)
- `rctd run` — full deconvolution pipeline, writes annotated h5ad output

The `run` command uses `RCTD` class directly (not `run_rctd()`), manages its own data loading, and writes results back into a copy of the spatial AnnData with `_write_results_to_adata()`. Progress goes to stderr when `--json` is set.

Tests in `tests/test_cli.py` — slow tests (marked `@pytest.mark.slow`) run actual deconvolution on synthetic data via `conftest._make_synthetic_reference` / `_make_synthetic_spatial`.

## Testing

```bash
uv run pytest tests/ -v
```

Tests use `torch.compile(dynamic=True)` which has a ~60s JIT warmup on first run.

### Test markers

- No marker: fast unit tests (~90s total including JIT warmup)
- `@pytest.mark.slow`: CLI integration tests that run full RCTD pipeline (~30s each)
- `@pytest.mark.performance`: benchmarking tests (excluded by default via `addopts`)

### Known tolerance notes

- `test_batch_matches_single` uses `atol=5e-5` — batch vs single-pixel IRWLS can differ slightly due to floating-point convergence order

## Benchmarking

GPU benchmarks are in `benchmarks/`. Submit via SLURM:

```bash
sbatch benchmarks/sbatch_bench_compare.sh   # baseline vs optimized comparison
sbatch benchmarks/sbatch_bench_gpu.sh        # optimized-only timing
```

## Autoresearch

Autonomous optimization framework inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The agent modifies solver code, benchmarks, keeps/discards based on speed + correctness.

### How to run

1. Read `benchmarks/program.md` for full instructions
2. Create branch: `git checkout -b autoresearch/<tag>`
3. Establish baseline: run benchmark as-is
4. Loop: modify → commit → benchmark → keep/discard

### Key rules

- **Metric**: `elapsed_s` (lower is better)
- **Correctness**: `weights_hash` must match baseline — if it changes, discard
- **Files to modify**: `src/rctd/_irwls.py`, `src/rctd/_likelihood.py`, `src/rctd/_simplex.py`
- **Read-only**: `benchmarks/bench_gpu.py`, `tests/`
- **Log results** to `results.tsv` (tab-separated)
- **GPU partition**: `--partition=GPU` (uppercase), servers fgcz-r-023 (L40S) and fgcz-c-056 (Blackwell)

### Profiling hot spots

From CPU profiling (2k pixels): `calc_q_all` 36%, QP solver 27%, eigh (PSD) 14%, bmm 9%.
GPU profile may differ — run `torch.profiler` to identify GPU-specific bottlenecks.
