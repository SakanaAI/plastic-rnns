"""Profile one-shot episodes and plot accuracy/latency Pareto frontiers.

The recurrent-cell configurations match the parameter-matched efficiency
comparison in the paper.  Each timed call resets the recurrent state once and
then processes a stateful sequence of five support and five query embeddings.
The embeddings are synthetic because the convolutional encoder is shared by
the recurrent architectures and is outside the comparison in Appendix A.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import seaborn as sns
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.efficiency.profile_models import (  # noqa: E402
    MODEL_SPECS,
    PARAM_MATCHED_CONFIGS,
    _expand_state,
)


# Across-all-five-query accuracy and 95% confidence intervals from
# Table "5-way 1-shot classification accuracy" in main_post_submission.tex.
PERFORMANCE = {
    "RNN": {"cifar100": (47.2, 1.3), "miniimagenet": (43.8, 1.2)},
    "PlasticRNN": {"cifar100": (47.8, 0.8), "miniimagenet": (45.4, 1.1)},
    "PlasticRNN (Oja)": {"cifar100": (48.1, 1.9), "miniimagenet": (45.8, 1.2)},
    "LSTM": {"cifar100": (55.9, 1.7), "miniimagenet": (49.3, 0.7)},
    "PlasticLSTM": {"cifar100": (55.7, 1.1), "miniimagenet": (49.6, 1.0)},
    "PlasticLSTM (simple)": {"cifar100": (55.4, 1.0), "miniimagenet": (50.0, 0.4)},
    "PlasticLSTM (retroactive)": {"cifar100": (55.6, 0.9), "miniimagenet": (49.7, 0.9)},
    "HyperLSTM": {"cifar100": (52.5, 0.4), "miniimagenet": (46.6, 1.1)},
    "STPN": {"cifar100": (59.2, 0.9), "miniimagenet": (50.5, 1.0)},
    "FW": {"cifar100": (58.3, 0.9), "miniimagenet": (47.7, 1.0)},
    "GDN": {"cifar100": (59.3, 0.7), "miniimagenet": (37.1, 19.4)},
    "CTM": {"cifar100": (50.5, 1.0), "miniimagenet": (46.8, 1.3)},
    "CTM (no NLMs)": {"cifar100": (59.4, 0.7), "miniimagenet": (48.9, 0.7)},
    "CTM Hebbian": {"cifar100": (57.3, 1.5), "miniimagenet": (50.9, 0.7)},
    "Plastic CTM": {"cifar100": (60.0, 0.6), "miniimagenet": (51.5, 1.4)},
}

DISPLAY_NAMES = {
    "PlasticRNN": "Plastic RNN",
    "PlasticRNN (Oja)": "Plastic RNN\n(Oja)",
    "PlasticLSTM": "Plastic LSTM\n(No Mod.)",
    "PlasticLSTM (simple)": "Plastic LSTM\n(Simple Mod.)",
    "PlasticLSTM (retroactive)": "Plastic LSTM\n(Retro. Mod.)",
    "CTM (no NLMs)": "Plastic CTM\n(No NLMs)",
    "CTM Hebbian": "Plastic CTM\n(Hebbian)",
    "Plastic CTM": "Plastic CTM\n(Ours)",
}

# Match the palette in analysis/maze_exploration/analyze.py.
_RL_PALETTE = sns.color_palette("hls", 6)
BASE_MODEL_COLORS = {
    "CTM": _RL_PALETTE[0],
    "LSTM": _RL_PALETTE[4],
    "RNN": _RL_PALETTE[2],
    "FW": _RL_PALETTE[3],
    "GDN": "#666666",
    "HyperLSTM": _RL_PALETTE[1],
    "STPN": _RL_PALETTE[5],
}

MODEL_MARKERS = {
    "RNN": "o",
    "PlasticRNN": "s",
    "PlasticRNN (Oja)": "D",
    "LSTM": "o",
    "PlasticLSTM": "s",
    "PlasticLSTM (simple)": "D",
    "PlasticLSTM (retroactive)": "^",
    "HyperLSTM": "P",
    "STPN": "v",
    "FW": "P",
    "GDN": "X",
    "CTM": "o",
    "CTM (no NLMs)": "s",
    "CTM Hebbian": "D",
    "Plastic CTM": "*",
}

CSV_FIELDS = [
    "model",
    "display_name",
    "d_model",
    "batch_size",
    "sequence_length",
    "run",
    "params",
    "episode_fwd_ms",
    "state_mem_kb_per_sample",
]


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _base_model(model: str) -> str:
    if "CTM" in model:
        return "CTM"
    if "LSTM" in model:
        return "HyperLSTM" if model == "HyperLSTM" else "LSTM"
    if "RNN" in model:
        return "RNN"
    return model


def _timed_calls(fn, device: str, n_warmup: int, n_measure: int) -> list[float]:
    for _ in range(n_warmup):
        fn()
    _sync(device)

    times = []
    for _ in range(n_measure):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - start) * 1_000)
    return times


def profile_episode(
    cell: torch.nn.Module,
    d_input: int,
    batch_size: int,
    sequence_length: int,
    device: str,
    n_warmup: int,
    n_measure: int,
) -> tuple[float, float]:
    """Return mean episode latency and auxiliary-state KB per sample."""
    cell = cell.to(device).eval()
    inputs = torch.randn(sequence_length, batch_size, d_input, device=device)

    def fresh_state():
        return _expand_state(cell.initialize_state(), batch_size, device)

    @torch.inference_mode()
    def forward_episode():
        state = fresh_state()
        for step in range(sequence_length):
            _, state = cell(inputs[step], state)

    forward_episode()
    times = _timed_calls(forward_episode, device, n_warmup, n_measure)

    state = fresh_state()
    state_floats = sum(t.numel() // max(t.shape[0], 1) for t in state)
    state_mem_kb = state_floats * 4 / 1024
    return mean(times), state_mem_kb


def run_profiling(args: argparse.Namespace) -> list[dict[str, object]]:
    specs = {name: (cell_class, kwargs_fn) for name, cell_class, kwargs_fn in MODEL_SPECS}
    rows: list[dict[str, object]] = []

    for name in PERFORMANCE:
        d_model = PARAM_MATCHED_CONFIGS[name]
        cell_class, kwargs_fn = specs[name]
        for run in range(1, args.n_runs + 1):
            kwargs = kwargs_fn(d_model)
            cell = cell_class(**kwargs)
            latency_ms, state_mem_kb = profile_episode(
                cell=cell,
                d_input=kwargs["d_input"],
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                device=args.device,
                n_warmup=args.n_warmup,
                n_measure=args.n_measure,
            )
            row = {
                "model": name,
                "display_name": DISPLAY_NAMES.get(name, name),
                "d_model": d_model,
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "run": run,
                "params": sum(parameter.numel() for parameter in cell.parameters()),
                "episode_fwd_ms": latency_ms,
                "state_mem_kb_per_sample": state_mem_kb,
            }
            rows.append(row)
            print(
                f"{name:<27s} run={run:<2d} params={row['params']:>7,d} "
                f"episode={latency_ms:>7.3f} ms"
            )
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    return rows


def write_profile(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_profile(input_path: Path) -> list[dict[str, object]]:
    numeric_ints = {"d_model", "batch_size", "sequence_length", "run", "params"}
    numeric_floats = {"episode_fwd_ms", "state_mem_kb_per_sample"}
    with input_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in numeric_ints:
            row[field] = int(row[field])
        for field in numeric_floats:
            row[field] = float(row[field])
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary = {}
    for model in PERFORMANCE:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            raise ValueError(f"No timing rows found for {model}")
        values = [float(row["episode_fwd_ms"]) for row in model_rows]
        summary[model] = {
            "latency_mean": mean(values),
            "latency_std": stdev(values) if len(values) > 1 else 0.0,
            "state_mem_kb": float(model_rows[0]["state_mem_kb_per_sample"]),
        }
    return summary


def pareto_models(summary: dict[str, dict[str, float]], dataset: str) -> list[str]:
    """Return models on the lower-latency/higher-accuracy frontier."""
    ordered = sorted(PERFORMANCE, key=lambda model: summary[model]["latency_mean"])
    frontier = []
    best_accuracy = float("-inf")
    for model in ordered:
        accuracy = PERFORMANCE[model][dataset][0]
        if accuracy > best_accuracy:
            frontier.append(model)
            best_accuracy = accuracy
    return frontier


def plot_tradeoff(
    summary: dict[str, dict[str, float]],
    output_stem: Path,
) -> None:
    sns.set_style("darkgrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Match the rendered width and compact panel height of rho_combined.pdf.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    datasets = [("cifar100", "CIFAR-100"), ("miniimagenet", "MiniImageNet")]
    for axis, (dataset, title) in zip(axes, datasets):
        frontier = pareto_models(summary, dataset)
        frontier_x = [summary[model]["latency_mean"] for model in frontier]
        frontier_y = [PERFORMANCE[model][dataset][0] for model in frontier]
        axis.plot(frontier_x, frontier_y, color="#777777", linestyle="--", linewidth=1.2, zorder=1)

        for model in PERFORMANCE:
            accuracy, accuracy_ci = PERFORMANCE[model][dataset]
            latency = summary[model]["latency_mean"]
            latency_std = summary[model]["latency_std"]
            color = BASE_MODEL_COLORS[_base_model(model)]
            is_ours = model == "Plastic CTM"
            marker = MODEL_MARKERS[model]
            size = 85 if is_ours else 40

            axis.errorbar(
                latency,
                accuracy,
                xerr=latency_std,
                yerr=accuracy_ci,
                fmt="none",
                ecolor=color,
                elinewidth=0.75,
                alpha=0.55,
                capsize=1.8,
                zorder=2,
            )
            axis.scatter(
                latency,
                accuracy,
                marker=marker,
                s=size,
                facecolor=color,
                edgecolor=color if not is_ours else "#8B1A1A",
                linewidth=1.5,
                zorder=3,
            )

        axis.set_title(title)
        axis.set_xlabel("Episode latency (ms)")
        axis.grid(True, alpha=0.3)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("Accuracy (%)")

    legend_handles = []
    for model in PERFORMANCE:
        color = BASE_MODEL_COLORS[_base_model(model)]
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                marker=MODEL_MARKERS[model],
                markersize=6.5 if model == "Plastic CTM" else 5,
                markerfacecolor=color,
                markeredgecolor=color if model != "Plastic CTM" else "#8B1A1A",
                linestyle="",
                label=DISPLAY_NAMES.get(model, model),
            )
        )
    fig.tight_layout(w_pad=2.0)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        frameon=False,
        fontsize=10,
        labelspacing=1.0,
        columnspacing=1.3,
        handletextpad=0.7,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = output_stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile 10-step recurrent episodes and plot one-shot accuracy versus latency"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-measure", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/efficiency/performance_cost"),
    )
    parser.add_argument(
        "--timings-csv",
        type=Path,
        help="Skip profiling and regenerate the plot from an existing raw timing CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sequence_length != 10:
        print(
            "Warning: one-shot episodes contain 10 steps; a different --sequence-length "
            "changes the interpretation of the x-axis."
        )

    if args.timings_csv:
        rows = read_profile(args.timings_csv)
    else:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        rows = run_profiling(args)
        timings_path = args.output_dir / "one_shot_episode_timings.csv"
        write_profile(rows, timings_path)
        print(f"Saved {timings_path}")

    summary = summarize(rows)
    plot_tradeoff(
        summary=summary,
        output_stem=args.output_dir / "one_shot_accuracy_vs_latency",
    )


if __name__ == "__main__":
    main()
