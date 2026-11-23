"""Plot FewShotCIFAR architecture-specific rho using the maze-RL line style."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ORDER = (
    "Plastic RNN",
    "Plastic RNN (Oja)",
    "Plastic LSTM (No Mod.)",
    "Plastic LSTM (Simple Mod.)",
    "Plastic LSTM (Retro. Mod.)",
    "Plastic CTM",
    "Plastic CTM (No NLMs)",
    "Plastic CTM (Hebbian)",
    "STPN",
)
MODEL_DISPLAY_NAMES = {
    "Plastic RNN": "Plastic RNN",
    "Plastic RNN (Oja)": "Plastic RNN\n(Oja)",
    "Plastic LSTM (No Mod.)": "Plastic LSTM\n(No Mod.)",
    "Plastic LSTM (Simple Mod.)": "Plastic LSTM\n(Simple Mod.)",
    "Plastic LSTM (Retro. Mod.)": "Plastic LSTM\n(Retro. Mod.)",
    "Plastic CTM": "Plastic CTM",
    "Plastic CTM (No NLMs)": "Plastic CTM\n(No NLMs)",
    "Plastic CTM (Hebbian)": "Plastic CTM\n(Hebbian)",
    "FW": "FW",
    "STPN": "STPN",
}


def configure_style():
    sns.set_palette("hls")
    sns.set_style("darkgrid")
    palette = sns.color_palette("hls", 6)
    base_colors = {
        "CTM": palette[0],
        "LSTM": palette[4],
        "RNN": palette[2],
        "FW": "#8C564B",
        "STPN": palette[5],
    }
    colors = {
        "Plastic RNN": base_colors["RNN"],
        "Plastic RNN (Oja)": base_colors["RNN"],
        "Plastic LSTM (No Mod.)": base_colors["LSTM"],
        "Plastic LSTM (Simple Mod.)": base_colors["LSTM"],
        "Plastic LSTM (Retro. Mod.)": base_colors["LSTM"],
        "Plastic CTM": base_colors["CTM"],
        "Plastic CTM (No NLMs)": base_colors["CTM"],
        "Plastic CTM (Hebbian)": base_colors["CTM"],
        "FW": base_colors["FW"],
        "STPN": base_colors["STPN"],
    }
    linestyles = {
        "Plastic RNN": "--",
        "Plastic RNN (Oja)": (0, (3, 1, 1, 1)),
        "Plastic LSTM (No Mod.)": "--",
        "Plastic LSTM (Simple Mod.)": (0, (3, 1, 1, 1)),
        "Plastic LSTM (Retro. Mod.)": ":",
        "Plastic CTM": "--",
        "Plastic CTM (No NLMs)": (0, (3, 1, 1, 1)),
        "Plastic CTM (Hebbian)": ":",
        "FW": (0, (5, 1)),
        "STPN": "-",
    }
    return colors, linestyles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/analysis/fewshot/plastic_ratio_cifar/"
            "plastic_ratio_by_iteration.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/fewshot/plastic_ratio_cifar"),
    )
    return parser.parse_args()


def load_curves(path: Path):
    rows_by_model = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows_by_model.setdefault(row["Model"], []).append(row)
    curves = {}
    for model, rows in rows_by_model.items():
        rows.sort(key=lambda row: int(row["Iteration"]))
        curves[model] = {
            "iterations": np.asarray([int(row["Iteration"]) for row in rows]),
            "mean": np.asarray([float(row["Mean ratio"]) for row in rows]),
            "std": np.asarray([float(row["Standard deviation"]) for row in rows]),
            "phases": [row["Phase"] for row in rows],
        }
    if not curves:
        raise ValueError(f"No rows found in {path}")
    return curves


def phase_spans(iterations, phases):
    spans = []
    start = 0
    for index in range(1, len(phases) + 1):
        if index == len(phases) or phases[index] != phases[start]:
            spans.append(
                (phases[start], int(iterations[start]), int(iterations[index - 1]))
            )
            start = index
    return spans


def plot_curves(
    curves,
    output_stem: Path,
    legend_position: str,
    output_aliases: tuple[Path, ...] = (),
) -> None:
    colors, linestyles = configure_style()
    plotted_models = [model for model in MODEL_ORDER if model in curves]
    if not plotted_models:
        raise ValueError("No FewShotCIFAR rho trajectories were found in the input")

    if legend_position not in {"below", "right", "none"}:
        raise ValueError(f"Unknown legend position: {legend_position}")
    figure_sizes = {"below": (16, 6), "right": (12, 6), "none": (8, 6)}
    figure_size = figure_sizes[legend_position]
    half_width = legend_position == "none"
    line_width = 2.5 if half_width else 2
    phase_fontsize = 18 if half_width else 16
    label_fontsize = 24 if half_width else 20
    tick_fontsize = 20 if half_width else 16
    figure, axis = plt.subplots(figsize=figure_size)

    handles = []
    for model in plotted_models:
        curve = curves[model]
        lower = np.maximum(0, curve["mean"] - curve["std"])
        upper = curve["mean"] + curve["std"]
        axis.plot(
            curve["iterations"],
            curve["mean"],
            linewidth=line_width,
            color=colors[model],
            linestyle=linestyles[model],
            alpha=0.7,
            zorder=3,
        )
        axis.fill_between(
            curve["iterations"],
            lower,
            upper,
            color=colors[model],
            alpha=0.2,
            linewidth=0,
            zorder=2,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=colors[model],
                linestyle=linestyles[model],
                linewidth=line_width,
                label=MODEL_DISPLAY_NAMES[model],
            )
        )

    reference = curves[plotted_models[0]]
    for phase, start, end in phase_spans(
        reference["iterations"], reference["phases"]
    ):
        if start > int(reference["iterations"][0]):
            axis.axvline(
                start - 0.5,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.28,
                zorder=1,
            )
        axis.text(
            (start + end) / 2,
            1.018,
            phase,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=phase_fontsize,
            color="black",
        )

    axis.set_xlim(
        float(reference["iterations"][0]), float(reference["iterations"][-1])
    )
    axis.set_ylim(bottom=0)
    axis.set_yscale("symlog", linthresh=0.01, linscale=0.8)
    axis.set_xlabel(r"$t$", fontsize=label_fontsize)
    axis.set_ylabel(r"$\rho^t$", fontsize=label_fontsize)
    axis.tick_params(axis="both", labelsize=tick_fontsize)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    if legend_position == "below":
        axis.legend(
            handles=handles,
            fontsize=16,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=6,
            frameon=False,
            labelspacing=1.0,
            columnspacing=1.5,
            handletextpad=0.7,
        )
    elif legend_position == "right":
        axis.legend(
            handles=handles,
            fontsize=16,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=2,
            frameon=False,
            labelspacing=1.2,
            columnspacing=1.2,
        )
    for stem in (output_stem, *output_aliases):
        stem.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def save_shared_legend(output_stem: Path) -> None:
    colors, linestyles = configure_style()
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[model],
            linestyle=linestyles[model],
            linewidth=2.5,
            label=MODEL_DISPLAY_NAMES[model],
        )
        for model in MODEL_ORDER
    ]
    # Matplotlib fills multi-column legends column-first. Interleave the two
    # desired rows so related variants read left-to-right in the paper.
    handles = [handle for pair in zip(handles[:5], handles[5:]) for handle in pair]
    figure = plt.figure(figsize=(8, 1.15))
    figure.legend(
        handles=handles,
        fontsize=9,
        loc="center",
        ncol=5,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.9,
        handletextpad=0.45,
        labelspacing=0.8,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )
    plt.close(figure)


def plot_csv(input_path: Path, output_stem: Path) -> None:
    curves = load_curves(input_path)
    tight_stem = output_stem.with_name(output_stem.name + "_tight_x")
    plot_curves(
        curves,
        output_stem,
        legend_position="below",
        output_aliases=(tight_stem,),
    )
    right_stem = output_stem.with_name(output_stem.name + "_legend_right")
    right_tight_stem = output_stem.with_name(
        output_stem.name + "_legend_right_tight_x"
    )
    plot_curves(
        curves,
        right_stem,
        legend_position="right",
        output_aliases=(right_tight_stem,),
    )
    half_width_stem = output_stem.with_name(
        output_stem.name + "_half_width_no_legend"
    )
    plot_curves(curves, half_width_stem, legend_position="none")
    save_shared_legend(output_stem.with_name("plastic_ratio_shared_legend"))


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else REPO_ROOT / args.output_dir
    )
    plot_csv(input_path, output_dir / "plastic_ratio_by_iteration")
    print(f"Wrote plot to {output_dir}")


if __name__ == "__main__":
    main()
