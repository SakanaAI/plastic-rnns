import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml


sns.set_palette("hls")
sns.set_style("darkgrid")

MODEL_ORDER = ["PlasticCTM", "PlasticCTMNoDynamicRescaling"]
MODEL_DISPLAY_NAMES = {
    "PlasticCTM": "Plastic CTM",
    "PlasticCTMNoDynamicRescaling": "Plastic CTM\n(No Dynamic Rescaling)",
}
MODEL_COLORS = {
    "PlasticCTM": sns.color_palette("hls", 2)[0],
    "PlasticCTMNoDynamicRescaling": sns.color_palette("hls", 2)[1],
}
MODEL_LINESTYLES = {
    "PlasticCTM": "-",
    "PlasticCTMNoDynamicRescaling": "--",
}
CHECKPOINT_PATTERN = re.compile(r"checkpoint_epoch_(\d+)\.pt$")
BETA_ORIGINAL_SUFFIX = "synapses.parametrizations.beta.original"
BETA_SUFFIX = "synapses.beta"


def normalize_dirs(dirs_config):
    normalized = {}
    for model_name, dirs in dirs_config.items():
        if isinstance(dirs, str):
            normalized[model_name] = [dirs]
        else:
            normalized[model_name] = dirs
    return normalized


def get_eval_return_curve(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    if "eval_return" not in metrics:
        return None

    steps = [point[0] for point in metrics["eval_return"]]
    returns = [point[1] for point in metrics["eval_return"]]
    return steps, returns


def collect_eval_return_curves(model_dirs):
    eval_return_curves = {model_name: [] for model_name in MODEL_ORDER}

    for model_name in MODEL_ORDER:
        for base_dir in model_dirs.get(model_name, []):
            if not os.path.exists(base_dir):
                print(f"Skipping missing directory: {base_dir}")
                continue

            run_ids = sorted(run_id for run_id in os.listdir(base_dir) if run_id.isdigit())
            for run_id in run_ids:
                curve = get_eval_return_curve(os.path.join(base_dir, run_id))
                if curve is not None:
                    eval_return_curves[model_name].append(curve)

    return eval_return_curves


def plot_eval_returns(eval_return_curves, output_dir):
    plt.figure(figsize=(8, 5))

    for model_name in MODEL_ORDER:
        curves = eval_return_curves[model_name]
        if not curves:
            print(f"No eval_return curves found for {model_name}")
            continue

        min_length = min(len(curve[1]) for curve in curves)
        steps = np.array(curves[0][0][:min_length]) * 512
        returns = np.array([curve[1][:min_length] for curve in curves])
        mean_returns = np.mean(returns, axis=0)
        std_returns = np.std(returns, axis=0)

        plt.plot(
            steps,
            mean_returns,
            label=MODEL_DISPLAY_NAMES[model_name],
            linewidth=2,
            color=MODEL_COLORS[model_name],
            linestyle=MODEL_LINESTYLES[model_name],
            alpha=0.85,
        )
        plt.fill_between(
            steps,
            mean_returns - std_returns,
            mean_returns + std_returns,
            color=MODEL_COLORS[model_name],
            alpha=0.2,
        )

    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel("Return", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, frameon=False)
    plt.gca().xaxis.get_offset_text().set_fontsize(12)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "plastic_ctm_no_dynamic_rescaling_eval_return.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "plastic_ctm_no_dynamic_rescaling_eval_return.pdf"), bbox_inches="tight")
    plt.close()


def checkpoint_epoch(path):
    match = CHECKPOINT_PATTERN.match(os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1))


def extract_beta(model_state_dict):
    for name, value in model_state_dict.items():
        if name.endswith(BETA_ORIGINAL_SUFFIX):
            return (2.0 * torch.sigmoid(value.detach().cpu())).item()
        if name.endswith(BETA_SUFFIX):
            return value.detach().cpu().item()
    return None


def get_beta_curve(run_dir):
    checkpoint_paths = []
    for name in os.listdir(run_dir):
        epoch = checkpoint_epoch(name)
        if epoch is not None:
            checkpoint_paths.append((epoch, os.path.join(run_dir, name)))

    if not checkpoint_paths:
        return None

    epochs, betas = [], []
    for epoch, checkpoint_path in sorted(checkpoint_paths):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        beta = extract_beta(checkpoint["model_state_dict"])
        if beta is not None:
            epochs.append(epoch)
            betas.append(beta)

    if not betas:
        return None

    return epochs, betas


def collect_beta_curves(model_dirs):
    beta_curves = []

    for base_dir in model_dirs.get("PlasticCTM", []):
        if not os.path.exists(base_dir):
            print(f"Skipping missing directory: {base_dir}")
            continue

        run_ids = sorted(run_id for run_id in os.listdir(base_dir) if run_id.isdigit())
        for run_id in run_ids:
            curve = get_beta_curve(os.path.join(base_dir, run_id))
            if curve is not None:
                beta_curves.append(curve)

    return beta_curves


def plot_beta_curve(beta_curves, output_dir):
    if not beta_curves:
        print("No beta curves found for PlasticCTM")
        return

    common_epochs = sorted(set.intersection(*(set(curve[0]) for curve in beta_curves)))
    if not common_epochs:
        print("No shared checkpoint epochs found for PlasticCTM beta curves")
        return

    beta_by_seed = []
    for epochs, betas in beta_curves:
        beta_by_epoch = dict(zip(epochs, betas))
        beta_by_seed.append([beta_by_epoch[epoch] for epoch in common_epochs])

    episodes = np.array(common_epochs) * 512
    betas = np.array(beta_by_seed)
    mean_betas = np.mean(betas, axis=0)
    std_betas = np.std(betas, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(
        episodes,
        mean_betas,
        label=MODEL_DISPLAY_NAMES["PlasticCTM"],
        linewidth=2,
        color=MODEL_COLORS["PlasticCTM"],
        alpha=0.85,
    )
    plt.fill_between(
        episodes,
        mean_betas - std_betas,
        mean_betas + std_betas,
        color=MODEL_COLORS["PlasticCTM"],
        alpha=0.2,
    )

    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel(r"$\beta$", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.get_offset_text().set_fontsize(12)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "plastic_ctm_beta_over_training.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "plastic_ctm_beta_over_training.pdf"), bbox_inches="tight")
    plt.close()


script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "experiment_dirs.yaml")
with open(config_path) as f:
    experiment_config = yaml.safe_load(f)

model_dirs = normalize_dirs(experiment_config["maze_exploration"])
eval_return_curves = collect_eval_return_curves(model_dirs)
plot_eval_returns(eval_return_curves, "outputs/analysis/maze_exploration")
beta_curves = collect_beta_curves(model_dirs)
plot_beta_curve(beta_curves, "outputs/analysis/maze_exploration")

print("Analysis complete! Plots saved to outputs/analysis/maze_exploration/")
