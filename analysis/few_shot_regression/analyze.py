import json
import os
import csv
import numpy as np
import yaml
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "experiment_dirs.yaml")
with open(config_path) as f:
    experiment_config = yaml.safe_load(f)


def normalize_dirs(dirs_config):
    normalized = {}
    for model_name, dirs in dirs_config.items():
        if isinstance(dirs, str):
            normalized[model_name] = [dirs]
        else:
            normalized[model_name] = dirs
    return normalized


def get_test_loss(run_dir):
    summary_path = f"{run_dir}/summary.json"
    if not os.path.exists(summary_path):
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    performance = summary.get("performance", None)
    if performance is None:
        return None
    return -performance


def analyze_group(model_dirs, output_name):
    results = {}
    for model_name, base_dirs in model_dirs.items():
        results[model_name] = []
        for base_dir in base_dirs:
            if os.path.exists(base_dir):
                run_ids = sorted([d for d in os.listdir(base_dir) if d.isdigit()])
                for run_id in run_ids:
                    loss = get_test_loss(f"{base_dir}/{run_id}")
                    if loss is not None:
                        results[model_name].append(loss)

    os.makedirs("outputs/analysis/few_shot_regression", exist_ok=True)
    csv_path = f"outputs/analysis/few_shot_regression/{output_name}_results.csv"

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['model', 'test_loss_mean', 'test_loss_std', 'test_loss_ci95']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for model_name in sorted(results.keys()):
            losses = results[model_name]
            if losses and len(losses) > 1:
                mean = float(np.mean(losses))
                std = float(np.std(losses, ddof=1))
                ci95 = float(stats.t.ppf(0.975, len(losses) - 1) * std / np.sqrt(len(losses)))
            elif losses:
                mean = float(losses[0])
                std = 0.0
                ci95 = 0.0
            else:
                mean = std = ci95 = float('nan')
            writer.writerow({
                'model': model_name,
                'test_loss_mean': mean,
                'test_loss_std': std,
                'test_loss_ci95': ci95,
            })
    print(f"Results written to {csv_path}")


for group_name, group_dirs in experiment_config.items():
    analyze_group(normalize_dirs(group_dirs), group_name)
