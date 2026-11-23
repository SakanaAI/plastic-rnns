"""Paired inference-time plasticity ablations for one-shot classification.

The enabled and disabled conditions use identical materialized test episodes.
The backbone is evaluated once per checkpoint, so only the recurrent
plasticity intervention differs between the two conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.fewshot.analyze_plastic_ratio import (  # noqa: E402
    DEFAULT_SEEDS,
    ENTRY_METADATA,
    batch_key,
    load_model,
    sample_test_batch,
    saved_accuracy,
)
from analysis.plasticity_ablation_utils import (  # noqa: E402
    encode_fewshot,
    paired_evaluation,
    summarize_paired,
)


MODEL_KEYS = (
    "PlasticRNN",
    "PlasticRNNOja",
    "PlasticLSTM",
    "PlasticLSTMSimple",
    "PlasticLSTMRetro",
    "STPN",
    "FW",
    "PlasticCTM",
    "CTMNoNLMs",
    "CTMHebbian",
)
METRICS = ("first_query", "all_queries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("experiment_dirs.yaml"),
    )
    parser.add_argument(
        "--dataset", choices=("CIFAR100", "MiniImageNet"), required=True
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples-per-seed", type=int, default=256)
    parser.add_argument("--evaluation-seed", type=int, default=12345)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--model",
        action="append",
        choices=MODEL_KEYS,
        help="Run only this experiment key; may be supplied more than once.",
    )
    parser.add_argument("--seed", action="append", type=int)
    args = parser.parse_args()
    if args.samples_per_seed < 1:
        parser.error("--samples-per-seed must be positive")
    return args


def experiment_entries(config_path: Path, dataset_name: str, selected: set[str]):
    section_name = (
        "fewshot_cifar" if dataset_name == "CIFAR100"
        else "fewshot_miniimagenet"
    )
    with config_path.open() as file:
        section = yaml.safe_load(file)[section_name]
    entries = []
    for key in MODEL_KEYS:
        if selected and key not in selected:
            continue
        directories = section[key]
        if isinstance(directories, str):
            directories = [directories]
        name, model_type = ENTRY_METADATA[key]
        for directory in directories:
            entries.append(
                {
                    "key": key,
                    "name": name,
                    "model_type": model_type,
                    "directory": directory,
                }
            )
    return entries


def query_indices(config) -> dict[str, list[int]]:
    ticks = int(config.task.iterations)
    support_ticks = (
        int(config.task.nKnovel) * int(config.task.nExemplars) * ticks
    )
    query_count = int(config.task.nTestNovel) + int(config.task.nTestBase)
    indices = [
        support_ticks + (query_index + 1) * ticks - 1
        for query_index in range(query_count)
    ]
    return {"first_query": indices[:1], "all_queries": indices}


def write_csv(results: list[dict], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Model", "Metric", "Enabled accuracy", "Disabled accuracy",
                "Absolute change (pp)", "Relative change (%)",
                "Mean seed relative change (%)", "Seed relative std (%)", "N",
            ]
        )
        for result in results:
            for metric in METRICS:
                values = result["summary"][metric]
                writer.writerow(
                    [
                        result["name"], metric, values["enabled_accuracy"],
                        values["disabled_accuracy"],
                        values["absolute_change_percentage_points"],
                        values["relative_change_percent"],
                        values["mean_seed_relative_change_percent"],
                        values["std_seed_relative_change_percent"],
                        values["enabled_total"],
                    ]
                )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    seeds = tuple(dict.fromkeys(args.seed or DEFAULT_SEEDS))
    entries = experiment_entries(
        args.config.resolve(), args.dataset, set(args.model or [])
    )
    output_dir = args.output_dir or Path(
        "outputs/analysis/fewshot/plasticity_ablation_"
        + ("cifar" if args.dataset == "CIFAR100" else "miniimagenet")
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_cache = {}
    results = []
    for entry in entries:
        print(f"{entry['name']} ({entry['key']})", flush=True)
        run_directory = REPO_ROOT / entry["directory"]
        seed_results = []
        intervention = None
        for seed in seeds:
            seed_directory = run_directory / str(seed)
            model, config, checkpoint_path = load_model(
                seed_directory,
                entry["model_type"],
                dataset_name=args.dataset,
                device=args.device,
            )
            configured_dataset = config.task.get("dataset", "CIFAR100")
            if configured_dataset != args.dataset:
                raise ValueError(
                    f"{seed_directory} is configured for {configured_dataset}, "
                    f"not {args.dataset}"
                )
            key = batch_key(config, args.samples_per_seed, args.evaluation_seed)
            if key not in batch_cache:
                batch_cache[key] = sample_test_batch(
                    config, args.samples_per_seed, args.evaluation_seed
                )
            embeddings, targets = encode_fewshot(model, batch_cache[key])
            enabled, disabled, current_intervention = paired_evaluation(
                model,
                entry["model_type"],
                embeddings,
                targets,
                query_indices(config),
            )
            if intervention is None:
                intervention = current_intervention
            elif intervention != current_intervention:
                raise RuntimeError("Intervention changed across seeds")
            seed_result = {
                "seed": seed,
                "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
                "saved_enabled_accuracy": saved_accuracy(seed_directory),
                "enabled": enabled,
                "disabled": disabled,
            }
            seed_results.append(seed_result)
            print(
                "  seed " + str(seed) + ": " + ", ".join(
                    f"{metric} {100 * enabled[metric]['correct'] / enabled[metric]['total']:.1f}%"
                    f" -> {100 * disabled[metric]['correct'] / disabled[metric]['total']:.1f}%"
                    for metric in METRICS
                ),
                flush=True,
            )
            del model, embeddings
            if args.device == "cuda":
                torch.cuda.empty_cache()

        summary = summarize_paired(seed_results, METRICS)
        results.append(
            {
                **entry,
                "intervention": intervention,
                "seeds": seed_results,
                "summary": summary,
            }
        )
        print(
            "  pooled relative change: " + ", ".join(
                f"{metric}={summary[metric]['relative_change_percent']:+.1f}%"
                for metric in METRICS
            ),
            flush=True,
        )

    payload = {
        "protocol": {
            "dataset": args.dataset,
            "device": args.device,
            "seeds": list(seeds),
            "samples_per_seed": args.samples_per_seed,
            "evaluation_seed": args.evaluation_seed,
            "pairing": "enabled and disabled conditions use identical materialized episodes",
            "aggregation": "pooled correct/total counts over seeds and episodes",
            "reported_change": "100 * (disabled - enabled) / enabled",
            "state_reset": "at the beginning of every episode",
            "excluded": "GDN has no comparable additive W + P^t decomposition",
        },
        "models": results,
    }
    (output_dir / "plasticity_ablation_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    write_csv(results, output_dir / "plasticity_ablation_summary.csv")
    print(f"Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
