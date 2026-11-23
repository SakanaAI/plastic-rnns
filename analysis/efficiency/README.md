# Efficiency Profiling

Profile all recurrent cell types for wall-clock time, parameter count, and peak GPU memory. Saves raw per-run results and mean/std summaries across runs to CSV.

## Usage

```bash
# Scaling sweep across d_model values
uv run python analysis/efficiency/profile_models.py --device cuda --d-models 64 128 256

# Parameter-matched
uv run python analysis/efficiency/profile_models.py --device cuda --param-matched

# One-shot classification accuracy versus 10-step recurrent episode latency
uv run python analysis/efficiency/plot_performance_cost.py --device cuda
```

For `--param-matched`, outputs are written to:

- `outputs/analysis/efficiency/profile_param_matched.csv` — one row per model/config/run
- `outputs/analysis/efficiency/profile_param_matched_summary.csv` — mean/std across runs

## Options

- `--device` — `cuda` or `cpu` (default: auto-detect)
- `--param-matched` — use parameter-matched d_model values
- `--d-models` — d_model values for scaling sweep (default: `64 128 256`)
- `--batch-sizes` — batch sizes to profile (default: `1 64`)
- `--n-runs` — independent profiling runs per model/config (default: `3`)
- `--n-warmup` — warmup iterations (default: `10`)
- `--n-measure` — measurement iterations (default: `100`)
- `--output-dir` — output directory (default: `outputs/analysis/efficiency`)

## Performance--cost plot

`plot_performance_cost.py` profiles a complete stateful 10-step episode (five
support and five query embeddings) for each parameter-matched recurrent cell.
It combines the resulting H100 latency with the across-all-query CIFAR-100 and
MiniImageNet accuracies reported in the paper, draws the Pareto frontiers, and
writes:

- `outputs/analysis/efficiency/performance_cost/one_shot_episode_timings.csv`
- `outputs/analysis/efficiency/performance_cost/one_shot_accuracy_vs_latency.pdf`
- `outputs/analysis/efficiency/performance_cost/one_shot_accuracy_vs_latency.png`

The default command uses batch size 64, 5 independent profiling runs, 20 warmup
episodes, and 100 measured episodes. To regenerate the plot without profiling:

```bash
uv run python analysis/efficiency/plot_performance_cost.py \
  --timings-csv outputs/analysis/efficiency/performance_cost/one_shot_episode_timings.csv
```
