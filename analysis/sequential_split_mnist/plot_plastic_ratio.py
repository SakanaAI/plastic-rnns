"""Plot Sequential Split-MNIST plastic-weight ratios in the RL plot style.

This is a plot-only entry point: it reads the per-iteration CSV written by
``analyze_plastic_ratio.py`` and recreates the PDF/PNG without loading any
models. Colors, line styles, confidence bands, typography, grid, and legend
placement follow ``analysis/maze_exploration/analyze.py``.

Example:
    uv run python analysis/sequential_split_mnist/plot_plastic_ratio.py
"""

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

# Keep the paper-facing plot restricted to architectures whose additive plastic
# component forms an effective matrix G^t = W + P^t. Non-plastic baselines are
# identically zero under this metric. The full style table remains below so an
# all-model diagnostic can still be requested from the CLI.
PLASTIC_MODEL_ORDER = (
    "Plastic RNN",
    "Plastic RNN (Oja)",
    "Plastic LSTM (No Mod.)",
    "Plastic LSTM (Simple Mod.)",
    "Plastic LSTM (Retro. Mod.)",
    "Plastic CTM (Ours)",
    "Plastic CTM (No NLMs)",
    "Plastic CTM (Hebbian)",
    "STPN",
)
ALL_MODEL_ORDER = (
    "RNN",
    "Plastic RNN",
    "Plastic RNN (Oja)",
    "LSTM",
    "Plastic LSTM (No Mod.)",
    "Plastic LSTM (Simple Mod.)",
    "Plastic LSTM (Retro. Mod.)",
    "CTM",
    "Plastic CTM (Ours)",
    "Plastic CTM (No NLMs)",
    "Plastic CTM (Hebbian)",
    "FW",
    "GDN",
    "HyperLSTM",
    "STPN",
)
MODEL_DISPLAY_NAMES = {
    "RNN": "RNN",
    "Plastic RNN": "Plastic RNN",
    "Plastic RNN (Oja)": "Plastic RNN\n(Oja)",
    "LSTM": "LSTM",
    "Plastic LSTM (No Mod.)": "Plastic LSTM\n(No Mod.)",
    "Plastic LSTM (Simple Mod.)": "Plastic LSTM\n(Simple Mod.)",
    "Plastic LSTM (Retro. Mod.)": "Plastic LSTM\n(Retro. Mod.)",
    "CTM": "CTM",
    "Plastic CTM (Ours)": "Plastic CTM",
    "Plastic CTM (No NLMs)": "Plastic CTM\n(No NLMs)",
    "Plastic CTM (Hebbian)": "Plastic CTM\n(Hebbian)",
    "FW": "FW",
    "GDN": "GDN",
    "HyperLSTM": "HyperLSTM",
    "STPN": "STPN",
}
PHASE_DISPLAY_NAMES = {
    "Set A support": "Train A",
    "Set A query": "Test A",
    "Set B support": "Train B",
    "Set A recall": "Test A",
    "Set B query": "Test B",
}


def configure_style() -> tuple[dict[str, tuple], dict[str, object]]:
    """Return the exact family colors and variant line styles used for RL."""
    sns.set_palette("hls")
    sns.set_style("darkgrid")
    color_palette = sns.color_palette("hls", 6)
    base_model_colors = {
        "CTM": color_palette[0],
        "LSTM": color_palette[4],
        "RNN": color_palette[2],
        "FW": color_palette[3],
        "HyperLSTM": color_palette[1],
        "STPN": color_palette[5],
        "GDN": "#9467BD",
    }
    model_colors = {
        "RNN": base_model_colors["RNN"],
        "Plastic RNN": base_model_colors["RNN"],
        "Plastic RNN (Oja)": base_model_colors["RNN"],
        "LSTM": base_model_colors["LSTM"],
        "Plastic LSTM (No Mod.)": base_model_colors["LSTM"],
        "Plastic LSTM (Simple Mod.)": base_model_colors["LSTM"],
        "Plastic LSTM (Retro. Mod.)": base_model_colors["LSTM"],
        "CTM": base_model_colors["CTM"],
        "Plastic CTM (Ours)": base_model_colors["CTM"],
        "Plastic CTM (No NLMs)": base_model_colors["CTM"],
        "Plastic CTM (Hebbian)": base_model_colors["CTM"],
        "FW": base_model_colors["FW"],
        "GDN": base_model_colors["GDN"],
        "HyperLSTM": base_model_colors["HyperLSTM"],
        "STPN": base_model_colors["STPN"],
    }
    model_linestyles = {
        "RNN": "-",
        "Plastic RNN": "--",
        "Plastic RNN (Oja)": (0, (3, 1, 1, 1)),
        "LSTM": "-",
        "Plastic LSTM (No Mod.)": "--",
        "Plastic LSTM (Simple Mod.)": (0, (3, 1, 1, 1)),
        "Plastic LSTM (Retro. Mod.)": ":",
        "CTM": "-",
        "Plastic CTM (Ours)": "--",
        "Plastic CTM (No NLMs)": (0, (3, 1, 1, 1)),
        "Plastic CTM (Hebbian)": ":",
        "FW": "-",
        "GDN": "-",
        "HyperLSTM": "-",
        "STPN": "-",
    }
    return model_colors, model_linestyles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/analysis/sequential_split_mnist/plastic_ratio/"
            "plastic_ratio_by_iteration.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/sequential_split_mnist/plastic_ratio"),
    )
    parser.add_argument(
        "--yscale",
        choices=("symlog", "linear"),
        default="symlog",
    )
    parser.add_argument(
        "--include-all-models",
        action="store_true",
        help="Include GDN and non-plastic baselines (primarily a diagnostic view).",
    )
    return parser.parse_args()


def load_curves(path: Path) -> dict[str, dict[str, np.ndarray | list[str]]]:
    rows_by_model: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows_by_model.setdefault(row["Model"], []).append(row)

    curves = {}
    for model, rows in rows_by_model.items():
        rows.sort(key=lambda row: int(row["Iteration"]))
        curves[model] = {
            "iterations": np.asarray([int(row["Iteration"]) for row in rows]),
            "mean": np.asarray([float(row["Mean ratio"]) for row in rows]),
            "std": np.asarray(
                [float(row["Standard deviation"]) for row in rows]
            ),
            "phases": [row["Phase"] for row in rows],
        }
    if not curves:
        raise ValueError(f"No plastic-ratio rows found in {path}")
    return curves


def contiguous_phase_spans(
    iterations: np.ndarray, phases: list[str]
) -> list[tuple[str, int, int]]:
    spans = []
    start_index = 0
    for index in range(1, len(phases) + 1):
        if index == len(phases) or phases[index] != phases[start_index]:
            spans.append(
                (
                    phases[start_index],
                    int(iterations[start_index]),
                    int(iterations[index - 1]),
                )
            )
            start_index = index
    return spans


def plot_curves(
    curves: dict[str, dict[str, np.ndarray | list[str]]],
    output_stem: Path,
    yscale: str = "linear",
    include_all_models: bool = False,
    legend_position: str = "below",
    output_aliases: tuple[Path, ...] = (),
) -> None:
    model_colors, model_linestyles = configure_style()
    unknown_models = set(curves) - set(ALL_MODEL_ORDER)
    if unknown_models:
        raise ValueError(f"Missing plot styles for: {', '.join(sorted(unknown_models))}")

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
    legend_handles = []
    model_order = ALL_MODEL_ORDER if include_all_models else PLASTIC_MODEL_ORDER
    plotted_models = [model for model in model_order if model in curves]

    for model in plotted_models:
        curve = curves[model]
        iterations = curve["iterations"]
        mean = curve["mean"]
        standard_deviation = curve["std"]
        color = model_colors[model]
        linestyle = model_linestyles[model]
        axis.plot(
            iterations,
            mean,
            label=MODEL_DISPLAY_NAMES[model],
            linewidth=line_width,
            color=color,
            linestyle=linestyle,
            alpha=0.7,
            zorder=3,
        )
        axis.fill_between(
            iterations,
            np.maximum(0, mean - standard_deviation),
            mean + standard_deviation,
            color=color,
            alpha=0.2,
            linewidth=0,
            zorder=2,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                linewidth=line_width,
                label=MODEL_DISPLAY_NAMES[model],
            )
        )

    reference = curves[plotted_models[0]]
    phase_spans = contiguous_phase_spans(
        reference["iterations"], reference["phases"]
    )
    for phase, start, end in phase_spans:
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
            PHASE_DISPLAY_NAMES.get(phase, phase),
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=phase_fontsize,
            color="black",
        )

    if yscale == "symlog":
        axis.set_yscale("symlog", linthresh=0.01, linscale=0.8)
    axis.set_ylim(bottom=0)
    axis.set_xlim(
        float(reference["iterations"][0]), float(reference["iterations"][-1])
    )
    axis.set_xlabel(r"$t$", fontsize=label_fontsize)
    axis.set_ylabel(r"$\rho^t$", fontsize=label_fontsize)
    axis.tick_params(axis="both", labelsize=tick_fontsize)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()

    if legend_position == "below":
        axis.legend(
            handles=legend_handles,
            fontsize=16,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=4,
            frameon=False,
            labelspacing=1.0,
            columnspacing=1.5,
            handletextpad=0.7,
        )
    elif legend_position == "right":
        axis.legend(
            handles=legend_handles,
            fontsize=16,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            frameon=False,
            labelspacing=1.2,
        )
    for stem in (output_stem, *output_aliases):
        stem.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_csv(
    input_path: Path,
    output_stem: Path,
    yscale: str = "linear",
    include_all_models: bool = False,
) -> None:
    curves = load_curves(input_path)
    tight_stem = output_stem.with_name(output_stem.name + "_tight_x")
    plot_curves(
        curves,
        output_stem,
        yscale=yscale,
        include_all_models=include_all_models,
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
        yscale=yscale,
        include_all_models=include_all_models,
        legend_position="right",
        output_aliases=(right_tight_stem,),
    )
    half_width_stem = output_stem.with_name(
        output_stem.name + "_half_width_no_legend"
    )
    plot_curves(
        curves,
        half_width_stem,
        yscale=yscale,
        include_all_models=include_all_models,
        legend_position="none",
    )


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else REPO_ROOT / args.output_dir
    )
    plot_csv(
        input_path,
        output_dir / "plastic_ratio_by_iteration",
        yscale=args.yscale,
        include_all_models=args.include_all_models,
    )
    print(f"Wrote plot to {output_dir}")


if __name__ == "__main__":
    main()
