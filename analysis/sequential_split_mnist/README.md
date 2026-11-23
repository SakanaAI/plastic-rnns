# Sequential Split-MNIST analysis

Copy `experiment_dirs.example.yaml` to `experiment_dirs.yaml` and set the paths to your multirun directories before running these commands from the repository root.

Episodes contain 25 images in five phases: A support, A query, B support, A retention, and B query. A and B use independent random label permutations, and query labels are withheld. The stored `accuracy_forward` metric is Set B accuracy and `accuracy_backward` is Set A retention; the FWT/BWT differences are computed separately by the transfer evaluation below.

Checkpoint paths may contain either `SequentialSplitMNIST` or the earlier `ContinualLearningACL` task name. Model weights and saved episode settings are loaded from each run, including hidden widths fitted during parameter matching. New analysis outputs go under `outputs/analysis/sequential_split_mnist/`.

Evaluate GEM-style forward and backward transfer from the trained checkpoints with

```bash
uv run python analysis/sequential_split_mnist/evaluate_transfer_metrics.py --device cuda
```

This writes the per-seed results, summary statistics, and a LaTeX table to
`outputs/analysis/sequential_split_mnist/transfer_metrics.*`. The evaluation treats the
recurrent state as the within-episode learner state. It evaluates matched queries
from cloned states so that an evaluation phase does not change the state used for
later task phases. State evolution within each query phase is retained.

Measure the relative plastic-component magnitude throughout each continual-learning
episode with

```bash
uv run python analysis/sequential_split_mnist/analyze_plastic_ratio.py \
    --samples-per-seed 64 \
    --torch-threads 8 \
    --strict-accuracy
```

This is a CPU-only evaluation using all five seeds for each architecture. It plots
the mean and standard deviation of `||P^t||_F / ||W||_F`, writes
raw/per-iteration/per-phase CSV files, and checks the sampled immediate,
backward, and forward query accuracies against the test accuracies saved with
the runs. `P^t` is the scaled additive component in the implementation,
measured just after each recurrent update; non-plastic models use `P^t = 0` by
definition.

The plot can be regenerated from the saved per-iteration CSV, without loading
the checkpoints again, with

```bash
uv run python analysis/sequential_split_mnist/plot_plastic_ratio.py
```

This plot-only script follows the maze-RL line-plot style: the same HLS family
colors, plastic-variant line styles, dark grid, mean/standard-deviation bands,
and frameless legend below the axes. The default paper plot contains the
additive-plastic architectures, including FW; its comparatively large
unnormalised fast-weight magnitude determines the vertical scale. GDN is
omitted because its standalone fast-weight state has no paired fixed recurrent
matrix for the same relative-magnitude definition. Pass
`--include-all-models` to recreate the full diagnostic view, or
`--yscale linear` to use a linear vertical axis.

Generate bar-chart versions of the FWT/BWT and plasticity-disabled results with

```bash
uv run python analysis/plot_transfer_and_ablation_bars.py
```

The script reads the saved JSON summaries and verifies every recorded run
directory against the current few-shot and Split-MNIST `experiment_dirs.yaml`
files before plotting. It writes a transfer figure and separate, style-matched
few-shot and continual-learning plasticity-ablation figures as PDF and PNG files
in `outputs/analysis/paper_bar_charts/`.
