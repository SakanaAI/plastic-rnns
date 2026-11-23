# Paper analyses

Run these commands from the repository root after training. Analysis outputs are written under `outputs/analysis/`.

## Experiment paths

For each task you want to analyze, copy its `experiment_dirs.example.yaml` to `experiment_dirs.yaml` in the same directory and replace the placeholders with your run paths. For example:

```bash
cp analysis/fewshot/experiment_dirs.example.yaml analysis/fewshot/experiment_dirs.yaml
```

The examples point to multirun directories containing numbered run directories (`0/`, `1/`, ...), each with `.hydra/config.yaml`, `metrics.json`, and `summary.json`. Add an entry for each architecture and condition you trained. Checkpoint analyses also require `checkpoint_epoch_*.pt`; enable `checkpoint.enabled=true` when training. Local path manifests are ignored by Git.

## Results and figures

| Paper result | Command after setting experiment paths |
| --- | --- |
| Copy task curves and bars | `uv run python analysis/copy/analyze.py` |
| One-shot classification tables | `uv run python analysis/fewshot/analyze.py` |
| Few-shot regression tables | `uv run python analysis/few_shot_regression/analyze.py` |
| Maze learning curves | `uv run python analysis/maze_exploration/analyze.py` |
| Maze rescaling ablation and learned plasticity scale | `uv run python analysis/maze_exploration/analyze_plastic_ctm_ablation.py` |
| Split-MNIST accuracy and transfer metrics | `uv run python analysis/sequential_split_mnist/evaluate_transfer_metrics.py --device cpu` |
| CIFAR fast-to-slow weight magnitudes | `uv run python analysis/fewshot/analyze_plastic_ratio.py` |
| Split-MNIST fast-to-slow weight magnitudes | `uv run python analysis/sequential_split_mnist/analyze_plastic_ratio.py` |
| Combined magnitude figure (after both analyses) | `uv run python analysis/fewshot/plot_combined_plastic_ratio.py` |
| CIFAR-100 plasticity interventions | `uv run python analysis/fewshot/analyze_plasticity_ablation.py --dataset CIFAR100` |
| MiniImageNet plasticity interventions | `uv run python analysis/fewshot/analyze_plasticity_ablation.py --dataset MiniImageNet` |
| Split-MNIST plasticity interventions | `uv run python analysis/sequential_split_mnist/analyze_plasticity_ablation.py` |
| Transfer and ablation bars (after the above evaluations) | `uv run python analysis/plot_transfer_and_ablation_bars.py` |

Run both few-shot interventions before generating the combined ablation bars. See [Split-MNIST analysis](sequential_split_mnist/README.md) for the evaluation protocol.

## PCA trajectories

Provide checkpoint paths directly to extract and compare plastic weights and hidden states:

```bash
uv run python analysis/sequential_split_mnist/analyze.py \
  --checkpoints path/to/lstm/checkpoint_epoch_1000.pt \
                path/to/plastic_ctm/checkpoint_epoch_1000.pt \
  --no-gif --device cpu
```

`analysis/sequential_split_mnist/plot_model_grid.py` assembles the five-model PCA grid from the generated per-model directories. Run it with `--help` for its input paths.

## Computational cost

See [efficiency profiling](efficiency/README.md) for recurrent-cell latency, state memory, and the accuracy/latency figure. Profiling can run on CPU or CUDA; the paper's timings were measured on an H100. The accuracy/latency plot uses the paper's reported accuracies and freshly measured timings, or a supplied timing CSV.
