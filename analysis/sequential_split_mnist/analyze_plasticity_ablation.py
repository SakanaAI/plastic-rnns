"""Paired inference-time plasticity ablations on Sequential Split-MNIST."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.sequential_split_mnist import infer_model_type  # noqa: E402
from analysis.plasticity_ablation_utils import (  # noqa: E402
    encode_continual,
    paired_evaluation,
    summarize_paired,
)
from analysis.sequential_split_mnist.analyze_plastic_ratio import (  # noqa: E402
    DEFAULT_SEEDS,
    load_experiment_entries,
    load_model,
    load_saved_accuracies,
    sample_test_batch,
)


MODEL_NAMES = (
    "Plastic RNN",
    "Plastic RNN (Oja)",
    "Plastic LSTM (No Mod.)",
    "Plastic LSTM (Simple Mod.)",
    "Plastic LSTM (Retro. Mod.)",
    "STPN",
    "FW",
    "Plastic CTM (No NLMs)",
    "Plastic CTM (Hebbian)",
    "Plastic CTM (Ours)",
)
METRICS = ("immediate", "backward", "forward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("experiment_dirs.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/sequential_split_mnist/plasticity_ablation"),
    )
    parser.add_argument("--samples-per-seed", type=int, default=256)
    parser.add_argument("--evaluation-seed", type=int, default=12345)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--successful-seeds-per-model",
        type=int,
        help=(
            "For each model, select this many lowest-numbered seeds whose saved "
            "accuracy exceeds --success-threshold in every phase."
        ),
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.98,
        help="Minimum saved accuracy in every phase for successful-seed selection.",
    )
    parser.add_argument("--model", action="append", choices=MODEL_NAMES)
    parser.add_argument("--seed", action="append", type=int)
    args = parser.parse_args()
    if args.samples_per_seed < 1:
        parser.error("--samples-per-seed must be positive")
    if args.successful_seeds_per_model is not None and args.successful_seeds_per_model < 1:
        parser.error("--successful-seeds-per-model must be positive")
    if not 0 <= args.success_threshold <= 1:
        parser.error("--success-threshold must lie in [0, 1]")
    return args


def select_successful_seeds(
    run_directory: Path,
    candidate_seeds: tuple[int, ...],
    count: int,
    threshold: float,
) -> list[int]:
    selected = []
    for seed in candidate_seeds:
        seed_directory = run_directory / str(seed)
        try:
            saved = load_saved_accuracies(seed_directory)
        except FileNotFoundError:
            continue
        if all(saved[metric] >= threshold for metric in METRICS):
            selected.append(seed)
        if len(selected) == count:
            break
    return selected


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
    candidate_seeds = tuple(dict.fromkeys(args.seed or DEFAULT_SEEDS))
    selected = set(args.model or MODEL_NAMES)
    entries = load_experiment_entries(args.config.resolve(), selected)
    entries = [entry for entry in entries if entry["name"] in selected]
    output_dir = (
        args.output_dir if args.output_dir.is_absolute()
        else REPO_ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = {}
    results = []
    skipped = []
    for entry in entries:
        print(f"{entry['name']} ({entry.get('model_type', 'inferred')})", flush=True)
        run_directory = REPO_ROOT / entry["directory"]
        model_type = infer_model_type(run_directory)
        if args.successful_seeds_per_model is not None:
            seeds = select_successful_seeds(
                run_directory,
                candidate_seeds,
                args.successful_seeds_per_model,
                args.success_threshold,
            )
            if len(seeds) < args.successful_seeds_per_model:
                reason = (
                    f"found {len(seeds)} successful complete seed(s), need "
                    f"{args.successful_seeds_per_model}"
                )
                print(f"  skipping: {reason}", flush=True)
                skipped.append({"name": entry["name"], "reason": reason})
                continue
            print(f"  selected successful seeds: {seeds}", flush=True)
        else:
            seeds = list(candidate_seeds)
        seed_results = []
        intervention = None
        for seed in seeds:
            seed_directory = run_directory / str(seed)
            model, config, checkpoint_path = load_model(
                seed_directory, model_type, device=args.device
            )
            key = (
                config.task.get("dataset_mode", "split_mnist"),
                int(config.task.get("iterations", 1)),
                int(config.task.get("n_exemplars_per_class", 1)),
                int(config.task.get("queries_per_loss", 5)),
                args.samples_per_seed,
                args.evaluation_seed,
            )
            if key not in batches:
                batches[key] = sample_test_batch(
                    config, args.samples_per_seed, args.evaluation_seed
                )
            embeddings, targets, sequence_info = encode_continual(
                model, batches[key]
            )
            metric_indices = {
                "immediate": sequence_info["immediate_query_indices"],
                "backward": sequence_info["backward_query_indices"],
                "forward": sequence_info["forward_query_indices"],
            }
            enabled, disabled, current_intervention = paired_evaluation(
                model, model_type, embeddings, targets, metric_indices
            )
            if intervention is None:
                intervention = current_intervention
            elif intervention != current_intervention:
                raise RuntimeError("Intervention changed across seeds")
            seed_result = {
                "seed": seed,
                "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
                "saved_enabled_accuracy": load_saved_accuracies(seed_directory),
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
                "name": entry["name"],
                "directory": entry["directory"],
                "model_type": model_type,
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
            "dataset": "Sequential Split-MNIST",
            "device": args.device,
            "candidate_seeds": list(candidate_seeds),
            "successful_seeds_per_model": args.successful_seeds_per_model,
            "success_threshold": args.success_threshold,
            "samples_per_seed": args.samples_per_seed,
            "evaluation_seed": args.evaluation_seed,
            "pairing": "enabled and disabled conditions use identical materialized episodes",
            "aggregation": "pooled correct/total counts over seeds and episodes",
            "reported_change": "100 * (disabled - enabled) / enabled",
            "state_reset": "at the beginning of every episode, not between phases",
            "excluded": "GDN has no comparable additive W + P^t decomposition",
        },
        "models": results,
        "skipped_models": skipped,
    }
    (output_dir / "plasticity_ablation_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    write_csv(results, output_dir / "plasticity_ablation_summary.csv")
    print(f"Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
