import json
import os
import csv
import numpy as np
import yaml
from scipy import stats

# Load experiment directories from YAML config
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "experiment_dirs.yaml")
with open(config_path) as f:
    experiment_config = yaml.safe_load(f)

# Normalize to lists (support both single dir string and list of dirs)
def normalize_dirs(dirs_config):
    normalized = {}
    for model_name, dirs in dirs_config.items():
        if isinstance(dirs, str):
            normalized[model_name] = [dirs]
        else:
            normalized[model_name] = dirs
    return normalized

cifar_dirs = normalize_dirs(experiment_config["fewshot_cifar"])
miniimagenet_dirs = normalize_dirs(experiment_config["fewshot_miniimagenet"])

def get_test_metrics(run_dir):
    """Get test accuracy and test accuracy_all_images from summary.json"""
    summary_path = f"{run_dir}/summary.json"
    if not os.path.exists(summary_path):
        return None, None

    with open(summary_path) as f:
        summary = json.load(f)

    test_metrics = summary.get("test_metrics", {})
    accuracy = test_metrics.get("accuracy", None)
    accuracy_all_images = test_metrics.get("accuracy_all_images", None)

    return accuracy, accuracy_all_images

def analyze_dataset(model_dirs, output_name):
    """Analyze a dataset and write results to CSV"""
    # Collect results
    results = {}
    for model_name, base_dirs in model_dirs.items():
        results[model_name] = {
            'accuracy': [],
            'accuracy_all_images': []
        }

        # Iterate through all directories for this model
        for base_dir in base_dirs:
            if os.path.exists(base_dir):
                # Automatically detect all run subdirectories (seeds)
                run_ids = sorted([d for d in os.listdir(base_dir) if d.isdigit()])
                for run_id in run_ids:
                    run_dir = f"{base_dir}/{run_id}"
                    accuracy, accuracy_all_images = get_test_metrics(run_dir)
                    if accuracy is not None:
                        results[model_name]['accuracy'].append(accuracy)
                    if accuracy_all_images is not None:
                        results[model_name]['accuracy_all_images'].append(accuracy_all_images)

    # Create output CSV
    os.makedirs("outputs/analysis/fewshot", exist_ok=True)
    csv_path = f"outputs/analysis/fewshot/{output_name}.csv"

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['model', 'test_accuracy_mean', 'test_accuracy_std', 'test_accuracy_ci95',
                      'test_accuracy_all_images_mean', 'test_accuracy_all_images_std', 'test_accuracy_all_images_ci95']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for model_name in sorted(results.keys()):
            accuracies = results[model_name]['accuracy']
            accuracies_all = results[model_name]['accuracy_all_images']

            # Calculate 95% confidence intervals
            if accuracies and len(accuracies) > 1:
                ci95 = stats.t.interval(0.95, len(accuracies)-1,
                                       loc=np.mean(accuracies),
                                       scale=stats.sem(accuracies))
                ci95_margin = ci95[1] - np.mean(accuracies)  # Half-width (margin of error)
            else:
                ci95_margin = np.nan

            if accuracies_all and len(accuracies_all) > 1:
                ci95_all = stats.t.interval(0.95, len(accuracies_all)-1,
                                           loc=np.mean(accuracies_all),
                                           scale=stats.sem(accuracies_all))
                ci95_all_margin = ci95_all[1] - np.mean(accuracies_all)  # Half-width (margin of error)
            else:
                ci95_all_margin = np.nan

            row = {
                'model': model_name,
                'test_accuracy_mean': np.mean(accuracies) if accuracies else np.nan,
                'test_accuracy_std': np.std(accuracies, ddof=1) if len(accuracies) > 1 else np.nan,  # Use sample std
                'test_accuracy_ci95': ci95_margin,
                'test_accuracy_all_images_mean': np.mean(accuracies_all) if accuracies_all else np.nan,
                'test_accuracy_all_images_std': np.std(accuracies_all, ddof=1) if len(accuracies_all) > 1 else np.nan,  # Use sample std
                'test_accuracy_all_images_ci95': ci95_all_margin
            }
            writer.writerow(row)

    print(f"Results written to {csv_path}")

# Analyze both datasets
analyze_dataset(cifar_dirs, "fewshot_cifar_results")
analyze_dataset(miniimagenet_dirs, "fewshot_miniimagenet_results")