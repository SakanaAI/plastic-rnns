"""Create paper-ready bar charts for transfer and plasticity ablations.

The script plots the JSON summaries written by ``evaluate_transfer_metrics.py``
and the two ``analyze_plasticity_ablation.py`` entry points.  Before plotting,
it verifies that every result came from the run directory currently selected in
the corresponding ``experiment_dirs.yaml`` file.

Run from the repository root with::

    uv run python analysis/plot_transfer_and_ablation_bars.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
PLASTIC_CTM_FEWSHOT_NAME = "Plastic CTM"
PLASTIC_CTM_DISPLAY_NAME = "Plastic CTM (Ours)"
TRANSFER_METRICS = (
    ("FWT", "fwt"),
    ("BWT", "bwt"),
)
ONE_SHOT_METRICS = (
    ("CIFAR-100", "cifar", "first_query"),
    ("MiniImageNet", "miniimagenet", "first_query"),
)
CONTINUAL_METRICS = (
    ("Set A Acc.", "continual", "immediate"),
    ("Set A Ret.", "continual", "backward"),
    ("Set B Acc.", "continual", "forward"),
)
MODEL_DISPLAY_NAMES = {
    "Plastic RNN (Oja)": "Plastic RNN\n(Oja)",
    "Plastic LSTM (No Mod.)": "Plastic LSTM\n(No Mod.)",
    "Plastic LSTM (Simple Mod.)": "Plastic LSTM\n(Simple Mod.)",
    "Plastic LSTM (Retro. Mod.)": "Plastic LSTM\n(Retro. Mod.)",
    "Plastic CTM (No NLMs)": "Plastic CTM\n(No NLMs)",
    "Plastic CTM (Hebbian)": "Plastic CTM\n(Hebbian)",
    "Plastic CTM (Ours)": "Plastic CTM\n(Ours)",
}
# The copy-task bars specify models from the bottom of the axis to the top.
COPY_BAR_MODEL_ORDER = (
    "Plastic CTM (Ours)",
    "Plastic CTM (Hebbian)",
    "Plastic CTM (No NLMs)",
    "CTM",
    "GDN",
    "FW",
    "STPN",
    "HyperLSTM",
    "Plastic LSTM (Retro. Mod.)",
    "Plastic LSTM (Simple Mod.)",
    "Plastic LSTM (No Mod.)",
    "LSTM",
    "Plastic RNN (Oja)",
    "Plastic RNN",
    "RNN",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-mnist-config",
        type=Path,
        default=Path("analysis/sequential_split_mnist/experiment_dirs.yaml"),
    )
    parser.add_argument(
        "--fewshot-config",
        type=Path,
        default=Path("analysis/fewshot/experiment_dirs.yaml"),
    )
    parser.add_argument(
        "--transfer-results",
        type=Path,
        default=Path("outputs/analysis/sequential_split_mnist/transfer_metrics.json"),
    )
    parser.add_argument(
        "--cifar-ablation-results",
        type=Path,
        default=Path(
            "outputs/analysis/fewshot/plasticity_ablation_cifar/"
            "plasticity_ablation_summary.json"
        ),
    )
    parser.add_argument(
        "--miniimagenet-ablation-results",
        type=Path,
        default=Path(
            "outputs/analysis/fewshot/plasticity_ablation_miniimagenet/"
            "plasticity_ablation_summary.json"
        ),
    )
    parser.add_argument(
        "--continual-ablation-results",
        type=Path,
        default=Path(
            "outputs/analysis/sequential_split_mnist/plasticity_ablation/"
            "plasticity_ablation_summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/paper_bar_charts"),
    )
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict:
    with repo_path(path).open() as file:
        return json.load(file)


def load_yaml(path: Path) -> dict:
    with repo_path(path).open() as file:
        return yaml.safe_load(file)


def require_matching_directory(
    *, result: dict, expected_directories: list[str], source: Path
) -> None:
    actual = result.get("directory")
    if actual not in expected_directories:
        raise ValueError(
            f"{source}: {result.get('name', '<unnamed>')} uses {actual!r}, but "
            f"the experiment config selects {expected_directories!r}"
        )


def canonical_ablation_name(result: dict) -> str:
    if (
        result.get("name") == PLASTIC_CTM_FEWSHOT_NAME
        and result.get("key") == "PlasticCTM"
    ):
        return PLASTIC_CTM_DISPLAY_NAME
    return result["name"]


def load_transfer_results(results_path: Path, config_path: Path) -> list[dict]:
    payload = load_json(results_path)
    config = load_yaml(config_path)
    by_name = {result["name"]: result for result in payload["models"]}
    ordered = []
    for entry in config["models"]:
        if entry["name"] not in by_name:
            continue
        result = by_name[entry["name"]]
        require_matching_directory(
            result=result,
            expected_directories=[entry["directory"]],
            source=repo_path(results_path),
        )
        ordered.append(result)
    if len(ordered) != len(by_name):
        unknown = sorted(set(by_name) - {result["name"] for result in ordered})
        raise ValueError(f"Transfer results contain models absent from config: {unknown}")
    return ordered


def load_fewshot_ablation_results(
    results_path: Path, config: dict, section: str
) -> dict[str, dict]:
    payload = load_json(results_path)
    results = {}
    for result in payload["models"]:
        configured = config[section][result["key"]]
        expected = configured if isinstance(configured, list) else [configured]
        require_matching_directory(
            result=result,
            expected_directories=expected,
            source=repo_path(results_path),
        )
        name = canonical_ablation_name(result)
        if name in results:
            raise ValueError(f"Duplicate {name!r} in {repo_path(results_path)}")
        results[name] = result
    return results


def load_continual_ablation_results(
    results_path: Path, config: dict
) -> dict[str, dict]:
    payload = load_json(results_path)
    configured = {entry["name"]: entry["directory"] for entry in config["models"]}
    results = {}
    for result in payload["models"]:
        if result["name"] not in configured:
            raise ValueError(
                f"{repo_path(results_path)} contains model absent from config: "
                f"{result['name']}"
            )
        require_matching_directory(
            result=result,
            expected_directories=[configured[result["name"]]],
            source=repo_path(results_path),
        )
        results[result["name"]] = result
    return results


def load_ablation_results(args: argparse.Namespace) -> tuple[list[str], dict]:
    fewshot_config = load_yaml(args.fewshot_config)
    split_config = load_yaml(args.split_mnist_config)
    datasets = {
        "cifar": load_fewshot_ablation_results(
            args.cifar_ablation_results, fewshot_config, "fewshot_cifar"
        ),
        "miniimagenet": load_fewshot_ablation_results(
            args.miniimagenet_ablation_results,
            fewshot_config,
            "fewshot_miniimagenet",
        ),
        "continual": load_continual_ablation_results(
            args.continual_ablation_results, split_config
        ),
    }
    model_sets = {dataset: set(results) for dataset, results in datasets.items()}
    reference = model_sets["continual"]
    for dataset, model_set in model_sets.items():
        if model_set != reference:
            raise ValueError(
                f"Ablation model mismatch for {dataset}: missing "
                f"{sorted(reference - model_set)}, extra {sorted(model_set - reference)}"
            )
    model_names = [
        entry["name"]
        for entry in split_config["models"]
        if entry["name"] in reference
    ]
    if set(model_names) != reference:
        raise ValueError("Could not order every ablation model from the Split-MNIST config")
    return model_names, datasets


def configure_style() -> None:
    # Match the copy-task bar plots in analysis/copy/analyze.py.
    sns.set_palette("hls")
    sns.set_style("darkgrid")


def copy_bar_model_order(model_names: list[str]) -> list[str]:
    available = set(model_names)
    ordered = [name for name in COPY_BAR_MODEL_ORDER if name in available]
    if set(ordered) != available:
        raise ValueError(
            "Models are missing from the copy-task display order: "
            f"{sorted(available - set(ordered))}"
        )
    return ordered


def save_figure(
    figure: plt.Figure, output_stem: Path, *, tight: bool = True
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_options = {"bbox_inches": "tight"} if tight else {}
    figure.savefig(output_stem.with_suffix(".pdf"), **save_options)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, **save_options)
    plt.close(figure)


def plot_grouped_bars(
    *,
    model_names: list[str],
    series: list[tuple[str, np.ndarray, np.ndarray]],
    x_label: str,
    legend_title: str,
    output_stem: Path,
    matched_ablation_layout: bool = False,
) -> None:
    figure_size = (11, 10) if matched_ablation_layout else (11, 14)
    figure, axis = plt.subplots(figsize=figure_size)
    y_positions = np.arange(len(model_names))
    colors = sns.color_palette("viridis", len(series))
    bar_height = 0.15
    y_offsets = (
        np.arange(len(series)) - (len(series) - 1) / 2
    ) * bar_height

    for series_index, ((_, values, deviations), color, y_offset) in enumerate(
        zip(series, colors, y_offsets, strict=True)
    ):
        axis.barh(
            y_positions + y_offset,
            values,
            bar_height,
            xerr=deviations,
            color=color,
            ecolor="black",
            alpha=1.0,
            zorder=series_index + 2,
            capsize=3,
        )

    axis.axvline(0, color="black", linewidth=1.2, zorder=len(series) + 2)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        [MODEL_DISPLAY_NAMES.get(name, name) for name in model_names],
        fontsize=20,
    )
    for label in axis.get_yticklabels():
        label.set_linespacing(1.4)
    axis.set_xlabel(x_label, fontsize=20)
    axis.set_ylabel("Model", fontsize=20, labelpad=15)
    axis.tick_params(axis="x", labelsize=16)
    axis.grid(True, alpha=0.3, axis="x")
    if matched_ablation_layout:
        axis.set_xlim(-92, 8)
        axis.set_xticks((-80, -60, -40, -20, 0))
        axis.set_ylim(-0.6, len(model_names) - 0.4)
        figure.subplots_adjust(left=0.29, right=0.98, top=0.98, bottom=0.18)
    else:
        plt.tight_layout()

    legend_handles = [
        Rectangle((0, 0), 1, 1, color=color, alpha=1.0) for color in colors
    ]
    legend_columns = len(series) if len(series) <= 3 else 3
    axis.legend(
        legend_handles,
        [label for label, _, _ in series],
        title=legend_title,
        fontsize=18,
        title_fontsize=18,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=legend_columns,
        columnspacing=3.5 if len(series) <= 3 else 1.5,
        frameon=False,
    )
    save_figure(figure, output_stem, tight=not matched_ablation_layout)


def plot_transfer(results: list[dict], output_stem: Path) -> None:
    by_name = {result["name"]: result for result in results}
    model_names = copy_bar_model_order(list(by_name))
    results = [by_name[name] for name in model_names]
    series = []
    for label, metric in TRANSFER_METRICS:
        means = np.asarray(
            [100 * result["summary"][metric]["mean"] for result in results]
        )
        deviations = np.asarray(
            [100 * result["summary"][metric]["std"] for result in results]
        )
        series.append((label, means, deviations))
    plot_grouped_bars(
        model_names=model_names,
        series=series,
        x_label="Transfer (percentage points)",
        legend_title="Metric",
        output_stem=output_stem,
    )


def metric_values(
    model_names: list[str], results: dict[str, dict], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    summaries = [results[name]["summary"][metric] for name in model_names]
    values = np.asarray([summary["relative_change_percent"] for summary in summaries])
    deviations = np.asarray(
        [summary["std_seed_relative_change_percent"] for summary in summaries]
    )
    return values, deviations


def plot_ablation_panels(
    model_names: list[str], datasets: dict, output_dir: Path
) -> None:
    model_names = copy_bar_model_order(model_names)
    fewshot_series = []
    for label, dataset, metric in ONE_SHOT_METRICS:
        values, deviations = metric_values(model_names, datasets[dataset], metric)
        fewshot_series.append((label, values, deviations))
    plot_grouped_bars(
        model_names=model_names,
        series=fewshot_series,
        x_label="Relative Accuracy Change (%)",
        legend_title="Dataset",
        output_stem=output_dir / "plasticity_ablation_fewshot_bars",
        matched_ablation_layout=True,
    )

    continual_series = []
    for label, dataset, metric in CONTINUAL_METRICS:
        values, deviations = metric_values(model_names, datasets[dataset], metric)
        continual_series.append((label, values, deviations))
    plot_grouped_bars(
        model_names=model_names,
        series=continual_series,
        x_label="Relative Accuracy Change (%)",
        legend_title="Phase",
        output_stem=output_dir / "plasticity_ablation_continual_bars",
        matched_ablation_layout=True,
    )


def main() -> None:
    args = parse_args()
    configure_style()
    output_dir = repo_path(args.output_dir)
    transfer_results = load_transfer_results(
        args.transfer_results, args.split_mnist_config
    )
    model_names, ablation_results = load_ablation_results(args)
    plot_transfer(transfer_results, output_dir / "transfer_metrics_bars")
    plot_ablation_panels(
        model_names,
        ablation_results,
        output_dir,
    )
    print(f"Wrote bar charts to {output_dir}")


if __name__ == "__main__":
    main()
