"""Plot the two paper rho panels with one shared legend."""

from __future__ import annotations

import argparse
import sys
from itertools import zip_longest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.fewshot.plot_plastic_ratio import (  # noqa: E402
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    configure_style,
    load_curves,
    phase_spans,
)
from analysis.sequential_split_mnist.plot_plastic_ratio import (  # noqa: E402
    PHASE_DISPLAY_NAMES,
)


RIGHT_MODEL_ORDER = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--left-input",
        type=Path,
        default=Path(
            "outputs/analysis/fewshot/plastic_ratio_cifar/"
            "plastic_ratio_by_iteration.csv"
        ),
    )
    parser.add_argument(
        "--right-input",
        type=Path,
        default=Path(
            "outputs/analysis/sequential_split_mnist/plastic_ratio/"
            "plastic_ratio_by_iteration.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/analysis/fewshot/plastic_ratio_cifar/"
            "plastic_ratio_combined"
        ),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def plot_panel(
    axis,
    curves,
    model_order,
    colors,
    linestyles,
    phase_display_names,
    yscale: str,
) -> None:
    plotted_models = [model for model in model_order if model in curves]
    if not plotted_models:
        raise ValueError("No requested rho trajectories were found")

    for model in plotted_models:
        curve = curves[model]
        lower = np.maximum(0, curve["mean"] - curve["std"])
        upper = curve["mean"] + curve["std"]
        axis.plot(
            curve["iterations"],
            curve["mean"],
            linewidth=2,
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
            phase_display_names.get(phase, phase),
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=14,
            color="black",
        )

    axis.set_xlim(
        float(reference["iterations"][0]),
        float(reference["iterations"][-1]),
    )
    axis.set_ylim(bottom=0)
    if yscale == "symlog":
        axis.set_yscale("symlog", linthresh=0.01, linscale=0.8)
    axis.set_xlabel(r"$t$", fontsize=18)
    axis.set_ylabel(r"$\rho^t$", fontsize=18)
    axis.tick_params(axis="both", labelsize=14)
    axis.grid(True, alpha=0.3)


def save_combined_plot(left_curves, right_curves, output_stem: Path) -> None:
    colors, linestyles = configure_style()
    if "Plastic CTM (Ours)" in right_curves:
        right_curves["Plastic CTM"] = right_curves.pop("Plastic CTM (Ours)")

    figure, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(16, 6))
    plot_panel(
        left_axis,
        left_curves,
        MODEL_ORDER,
        colors,
        linestyles,
        {},
        yscale="symlog",
    )
    plot_panel(
        right_axis,
        right_curves,
        RIGHT_MODEL_ORDER,
        colors,
        linestyles,
        PHASE_DISPLAY_NAMES,
        yscale="linear",
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=colors[model],
            linestyle=linestyles[model],
            linewidth=2,
            label=MODEL_DISPLAY_NAMES[model],
        )
        for model in MODEL_ORDER
    ]
    # Matplotlib fills columns first; interleave the desired top and bottom rows.
    handles = [
        handle
        for pair in zip_longest(handles[:5], handles[5:])
        for handle in pair
        if handle is not None
    ]

    figure.tight_layout(w_pad=2.0)
    figure.legend(
        handles=handles,
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
        labelspacing=1.0,
        columnspacing=1.3,
        handletextpad=0.7,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    save_combined_plot(
        load_curves(resolve(args.left_input)),
        load_curves(resolve(args.right_input)),
        resolve(args.output),
    )
    print(f"Wrote combined plot to {resolve(args.output)}")


if __name__ == "__main__":
    main()
