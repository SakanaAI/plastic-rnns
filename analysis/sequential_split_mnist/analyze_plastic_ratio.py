"""Measure the plastic-to-fixed weight ratio on Sequential Split-MNIST.

For every checkpoint seed and sampled test episode, this script records

    ||P^t||_F / ||W||_F,

where ``P^t`` is the *scaled additive component* represented by the recurrent
state immediately after iteration ``t``. Thus it is the plastic component
available to the next recurrent update, rather than an unscaled Hebbian or
synchronization trace. Non-plastic architectures use ``P^t = 0`` by
definition. HyperLSTM is included under that convention because its dynamic
modulation is multiplicative and has no additive ``P^t`` in this decomposition.

The script is deliberately CPU-only. It also evaluates the query predictions
made along the measured trajectories and compares their phase accuracies with
the test accuracies saved beside each checkpoint.

Example:
    uv run python analysis/sequential_split_mnist/analyze_plastic_ratio.py \
        --samples-per-seed 64 --torch-threads 8 --strict-accuracy
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.sequential_split_mnist import infer_model_type  # noqa: E402
from tasks.sequential_split_mnist import (  # noqa: E402
    SequentialSplitMNISTDataloader,
    SequentialSplitMNISTDataset,
    SequentialSplitMNISTModel,
)
from analysis.plastic_ratio_utils import (  # noqa: E402
    expand_state,
    plastic_component_norms,
)
from analysis.sequential_split_mnist.plot_plastic_ratio import plot_csv  # noqa: E402


DEFAULT_SEEDS = (0, 1, 2, 3, 4)
MODEL_ORDER = (
    "Plastic CTM (Ours)",
    "Plastic LSTM (No Mod.)",
    "Plastic LSTM (Simple Mod.)",
    "Plastic LSTM (Retro. Mod.)",
    "CTM",
    "LSTM",
    "RNN",
    "Plastic RNN",
    "Plastic RNN (Oja)",
    "FW",
    "GDN",
    "HyperLSTM",
    "STPN",
    "Plastic CTM (No NLMs)",
    "Plastic CTM (Hebbian)",
)
ACCURACY_PHASES = (
    ("immediate", "accuracy_immediate"),
    ("backward", "accuracy_backward"),
    ("forward", "accuracy_forward"),
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("experiment_dirs.yaml"),
        help="YAML file containing model names and multirun directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/sequential_split_mnist/plastic_ratio"),
    )
    parser.add_argument("--samples-per-seed", type=int, default=64)
    parser.add_argument("--evaluation-seed", type=int, default=12345)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument(
        "--accuracy-tolerance",
        type=float,
        default=0.15,
        help="Maximum allowed absolute sampled-vs-saved accuracy difference.",
    )
    parser.add_argument(
        "--strict-accuracy",
        action="store_true",
        help="Exit unsuccessfully if an aggregate accuracy check exceeds the tolerance.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Analyze only the named model. May be supplied more than once.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Candidate checkpoint seed. May be supplied more than once (default: 0--4).",
    )
    parser.add_argument(
        "--successful-seeds-per-model",
        type=int,
        help=(
            "Use the first N candidate seeds whose three saved phase accuracies "
            "all meet --success-threshold."
        ),
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.99,
        help="Minimum saved accuracy in every phase for successful-seed selection.",
    )
    args = parser.parse_args()
    if args.samples_per_seed < 1:
        parser.error("--samples-per-seed must be positive")
    if args.accuracy_tolerance < 0:
        parser.error("--accuracy-tolerance must be non-negative")
    if (
        args.successful_seeds_per_model is not None
        and args.successful_seeds_per_model < 1
    ):
        parser.error("--successful-seeds-per-model must be positive")
    if not 0 <= args.success_threshold <= 1:
        parser.error("--success-threshold must be between zero and one")
    return args


def find_checkpoint(seed_directory: Path) -> Path:
    checkpoints = list(seed_directory.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {seed_directory}")
    return max(checkpoints, key=lambda path: int(path.stem.rsplit("_", 1)[1]))


def load_model(
    seed_directory: Path,
    model_type: str,
    device: str | torch.device = "cpu",
):
    """Instantiate and load one checkpoint onto the requested device."""
    config = OmegaConf.load(seed_directory / ".hydra" / "config.yaml")
    checkpoint_path = find_checkpoint(seed_directory)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = dict(checkpoint["model_state_dict"])
    if state_dict and all(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
        }

    # The model size was selected during the parameter-matching search and is
    # encoded exactly in the checkpoint, whereas Hydra retains the search-space
    # starting value.
    config.model.d_model = int(state_dict["output_proj.weight"].shape[1])
    device = torch.device(device)
    model = SequentialSplitMNISTModel(
        model_cfg=config.model,
        task_cfg=config.task,
        model_type=model_type,
    ).to(device)
    with torch.inference_mode():
        model(
            torch.zeros(1, 1, 3, 32, 32, device=device),
            torch.zeros(1, 1, dtype=torch.long, device=device),
        )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, config, checkpoint_path


def phase_layout(config) -> tuple[list[str], list[tuple[str, int, int]]]:
    internal_ticks = int(config.task.get("iterations", 1))
    n_classes = int(config.task.get("out_dims", 5))
    n_support = n_classes * int(config.task.get("n_exemplars_per_class", 1))
    n_queries = int(config.task.get("queries_per_loss", 5))
    image_phases = (
        ("Set A support", n_support),
        ("Set A query", n_queries),
        ("Set B support", n_support),
        ("Set A recall", n_queries),
        ("Set B query", n_queries),
    )
    tick_phases: list[str] = []
    spans = []
    start = 1
    for name, image_count in image_phases:
        tick_count = image_count * internal_ticks
        tick_phases.extend([name] * tick_count)
        spans.append((name, start, start + tick_count - 1))
        start += tick_count
    return tick_phases, spans


def sample_test_batch(config, sample_count: int, evaluation_seed: int):
    random.seed(evaluation_seed)
    np.random.seed(evaluation_seed)
    torch.manual_seed(evaluation_seed)
    dataset = SequentialSplitMNISTDataset(
        phase="test",
        dataset_mode=config.task.get("dataset_mode", "split_mnist"),
    )
    dataloader = SequentialSplitMNISTDataloader(
        dataset=dataset,
        internal_ticks=config.task.get("iterations", 1),
        n_exemplars_per_class=config.task.get("n_exemplars_per_class", 1),
        queries_per_loss=config.task.get("queries_per_loss", 5),
        batch_size=sample_count,
        epoch_size=sample_count,
    )
    (inputs, aux_inputs), targets = next(iter(dataloader))
    return (
        inputs.transpose(0, 1).contiguous(),
        aux_inputs.transpose(0, 1).contiguous(),
        targets.transpose(0, 1).contiguous(),
        dataloader.sequence_info,
    )


def batch_cache_key(config, sample_count: int, evaluation_seed: int) -> tuple:
    return (
        config.task.get("dataset_mode", "split_mnist"),
        int(config.task.get("iterations", 1)),
        int(config.task.get("n_exemplars_per_class", 1)),
        int(config.task.get("queries_per_loss", 5)),
        sample_count,
        evaluation_seed,
    )


def evaluate_trajectories(model, model_type: str, batch):
    inputs, aux_inputs, targets, sequence_info = batch
    time_steps, batch_size, channels, height, width = inputs.shape
    cell = model.rnn.recurrent_module

    accuracy_indices = {
        "immediate": set(sequence_info["immediate_query_indices"]),
        "backward": set(sequence_info["backward_query_indices"]),
        "forward": set(sequence_info["forward_query_indices"]),
    }
    accuracy_counts = {
        phase: {"correct": 0, "total": 0} for phase in accuracy_indices
    }
    ratios = []
    plastic_norms = []
    fixed_norms = []
    component_definition = None

    with torch.inference_mode():
        flat_inputs = inputs.reshape(
            time_steps * batch_size, channels, height, width
        )
        flat_aux = aux_inputs.reshape(time_steps * batch_size)
        label_embeddings = torch.nn.functional.one_hot(
            flat_aux, num_classes=model.num_classes + 1
        ).float()
        embeddings = model.backbone(flat_inputs, label_embeddings).view(
            time_steps, batch_size, -1
        )
        state = expand_state(model.rnn.get_initial_state(), batch_size)

        for iteration in range(time_steps):
            hidden, state = cell(embeddings[iteration], state)
            p_norm, w_norm, definition = plastic_component_norms(
                model_type, cell, state, batch_size
            )
            if component_definition is None:
                component_definition = definition
            elif component_definition != definition:
                raise RuntimeError("P/W component definition changed during evaluation")
            ratios.append(p_norm / w_norm.clamp_min(torch.finfo(w_norm.dtype).eps))
            plastic_norms.append(p_norm)
            fixed_norms.append(w_norm)

            logits = model.output_proj(hidden)
            predictions = logits.argmax(dim=-1)
            for phase, indices in accuracy_indices.items():
                if iteration in indices:
                    accuracy_counts[phase]["correct"] += int(
                        (predictions == targets[iteration]).sum().item()
                    )
                    accuracy_counts[phase]["total"] += batch_size

    return {
        "ratio": torch.stack(ratios, dim=1).cpu().numpy(),
        "plastic_norm": torch.stack(plastic_norms, dim=1).cpu().numpy(),
        "fixed_norm": torch.stack(fixed_norms, dim=1).cpu().numpy(),
        "accuracy_counts": accuracy_counts,
        "component_definition": component_definition,
    }


def load_saved_accuracies(seed_directory: Path) -> dict[str, float]:
    with (seed_directory / "summary.json").open() as file:
        metrics = json.load(file)["test_metrics"]
    return {
        phase: float(metrics[metric_name])
        for phase, metric_name in ACCURACY_PHASES
    }


def load_experiment_entries(config_path: Path, selected_models: set[str]):
    with config_path.open() as file:
        entries = yaml.safe_load(file)["models"]
    if selected_models:
        known = {entry["name"] for entry in entries}
        unknown = selected_models - known
        if unknown:
            raise ValueError(f"Unknown model name(s): {', '.join(sorted(unknown))}")
        entries = [entry for entry in entries if entry["name"] in selected_models]
    order = {name: index for index, name in enumerate(MODEL_ORDER)}
    return sorted(entries, key=lambda entry: order.get(entry["name"], len(order)))


def summarize_accuracy(seed_results: list[dict], tolerance: float):
    summary = {}
    for phase, _ in ACCURACY_PHASES:
        correct = sum(
            result["accuracy_counts"][phase]["correct"] for result in seed_results
        )
        total = sum(
            result["accuracy_counts"][phase]["total"] for result in seed_results
        )
        sampled = correct / total
        saved_values = [result["saved_accuracy"][phase] for result in seed_results]
        expected = float(np.mean(saved_values))
        absolute_difference = abs(sampled - expected)
        summary[phase] = {
            "sampled": sampled,
            "sample_correct": correct,
            "sample_total": total,
            "saved_mean": expected,
            "saved_std_across_seeds": float(np.std(saved_values)),
            "absolute_difference": absolute_difference,
            "within_tolerance": absolute_difference <= tolerance,
        }
    return summary


def write_raw_csv(results: list[dict], tick_phases: list[str], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Model",
                "Seed",
                "Sample",
                "Iteration",
                "Phase",
                "Plastic norm",
                "Fixed norm",
                "Plastic/fixed ratio",
            ]
        )
        for model_result in results:
            for seed_result in model_result["seeds"]:
                for sample_index in range(seed_result["ratio"].shape[0]):
                    for tick_index, phase in enumerate(tick_phases):
                        writer.writerow(
                            [
                                model_result["name"],
                                seed_result["seed"],
                                sample_index,
                                tick_index + 1,
                                phase,
                                seed_result["plastic_norm"][sample_index, tick_index],
                                seed_result["fixed_norm"][sample_index, tick_index],
                                seed_result["ratio"][sample_index, tick_index],
                            ]
                        )


def write_iteration_summary_csv(
    results: list[dict], tick_phases: list[str], path: Path
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Model", "Iteration", "Phase", "Mean ratio", "Standard deviation", "N"]
        )
        for model_result in results:
            traces = model_result["traces"]
            for tick_index, phase in enumerate(tick_phases):
                values = traces[:, tick_index]
                writer.writerow(
                    [
                        model_result["name"],
                        tick_index + 1,
                        phase,
                        values.mean(),
                        values.std(),
                        len(values),
                    ]
                )


def write_phase_summary_csv(
    results: list[dict], tick_phases: list[str], path: Path
) -> None:
    ordered_phases = list(dict.fromkeys(tick_phases))
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Phase", "Mean ratio", "Standard deviation", "N"])
        for model_result in results:
            traces = model_result["traces"]
            for phase in ordered_phases:
                indices = [
                    index for index, tick_phase in enumerate(tick_phases)
                    if tick_phase == phase
                ]
                values = traces[:, indices].reshape(-1)
                writer.writerow(
                    [
                        model_result["name"],
                        phase,
                        values.mean(),
                        values.std(),
                        len(values),
                    ]
                )


def iter_accuracy_rows(results: list[dict]) -> Iterable[list]:
    for model_result in results:
        for phase, _ in ACCURACY_PHASES:
            value = model_result["accuracy_summary"][phase]
            yield [
                model_result["name"],
                "all",
                phase,
                value["sampled"],
                value["sample_correct"],
                value["sample_total"],
                value["saved_mean"],
                value["saved_std_across_seeds"],
                value["absolute_difference"],
                value["within_tolerance"],
            ]
        for seed_result in model_result["seeds"]:
            for phase, _ in ACCURACY_PHASES:
                counts = seed_result["accuracy_counts"][phase]
                sampled = counts["correct"] / counts["total"]
                saved = seed_result["saved_accuracy"][phase]
                difference = abs(sampled - saved)
                yield [
                    model_result["name"],
                    seed_result["seed"],
                    phase,
                    sampled,
                    counts["correct"],
                    counts["total"],
                    saved,
                    "",
                    difference,
                    difference <= model_result["accuracy_tolerance"],
                ]


def write_accuracy_csv(results: list[dict], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Model",
                "Seed",
                "Phase",
                "Sampled accuracy",
                "Sample correct",
                "Sample total",
                "Saved accuracy mean",
                "Saved accuracy std",
                "Absolute difference",
                "Within tolerance",
            ]
        )
        writer.writerows(iter_accuracy_rows(results))


def serializable_results(results: list[dict]) -> list[dict]:
    serialized = []
    for result in results:
        serialized.append(
            {
                "name": result["name"],
                "directory": result["directory"],
                "model_type": result["model_type"],
                "component_definition": result["component_definition"],
                "accuracy": result["accuracy_summary"],
                "seeds": [
                    {
                        "seed": seed["seed"],
                        "checkpoint": seed["checkpoint"],
                        "saved_accuracy": seed["saved_accuracy"],
                        "sampled_accuracy": {
                            phase: (
                                counts["correct"] / counts["total"]
                            )
                            for phase, counts in seed["accuracy_counts"].items()
                        },
                    }
                    for seed in result["seeds"]
                ],
            }
        )
    return serialized


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    device = torch.device("cpu")
    if device.type != "cpu":  # Defensive: this analysis must remain CPU-only.
        raise RuntimeError("Plastic-ratio analysis must run on CPU")

    config_path = args.config.resolve()
    selected_models = set(args.model or [])
    candidate_seeds = tuple(dict.fromkeys(args.seed or DEFAULT_SEEDS))
    entries = load_experiment_entries(config_path, selected_models)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_cache = {}
    results = []
    reference_tick_phases = None
    reference_phase_spans = None

    for entry in entries:
        run_directory = Path(entry["directory"])
        if not run_directory.is_absolute():
            run_directory = REPO_ROOT / run_directory
        model_type = entry.get("model_type", infer_model_type(run_directory))
        print(f"{entry['name']} ({model_type})", flush=True)
        seed_results = []
        component_definition = None

        seeds = candidate_seeds
        if args.successful_seeds_per_model is not None:
            successful_seeds = []
            for seed in candidate_seeds:
                saved_accuracy = load_saved_accuracies(run_directory / str(seed))
                if all(
                    value >= args.success_threshold
                    for value in saved_accuracy.values()
                ):
                    successful_seeds.append(seed)
            if len(successful_seeds) < args.successful_seeds_per_model:
                raise RuntimeError(
                    f"{entry['name']} has only {len(successful_seeds)} successful "
                    f"seed(s) at threshold {args.success_threshold:.3f}; "
                    f"need {args.successful_seeds_per_model}"
                )
            seeds = tuple(successful_seeds[: args.successful_seeds_per_model])
            print(f"  selected successful seeds: {', '.join(map(str, seeds))}")

        for seed in seeds:
            seed_directory = run_directory / str(seed)
            model, config, checkpoint_path = load_model(seed_directory, model_type)
            key = batch_cache_key(config, args.samples_per_seed, args.evaluation_seed)
            if key not in batch_cache:
                batch_cache[key] = sample_test_batch(
                    config, args.samples_per_seed, args.evaluation_seed
                )
            tick_phases, phase_spans = phase_layout(config)
            if reference_tick_phases is None:
                reference_tick_phases = tick_phases
                reference_phase_spans = phase_spans
            elif reference_tick_phases != tick_phases:
                raise ValueError("Selected runs do not share one episode phase layout")

            evaluated = evaluate_trajectories(
                model, model_type, batch_cache[key]
            )
            if component_definition is None:
                component_definition = evaluated["component_definition"]
            elif component_definition != evaluated["component_definition"]:
                raise RuntimeError("P/W definition differs across seeds")
            saved_accuracy = load_saved_accuracies(seed_directory)
            evaluated.update(
                {
                    "seed": seed,
                    "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
                    "saved_accuracy": saved_accuracy,
                }
            )
            seed_results.append(evaluated)
            sampled_text = ", ".join(
                f"{phase}={100 * counts['correct'] / counts['total']:.1f}%"
                for phase, counts in evaluated["accuracy_counts"].items()
            )
            print(f"  seed {seed}: {sampled_text}", flush=True)
            del model

        traces = np.concatenate([result["ratio"] for result in seed_results], axis=0)
        accuracy_summary = summarize_accuracy(seed_results, args.accuracy_tolerance)
        check_text = ", ".join(
            f"{phase} Δ={100 * values['absolute_difference']:.1f}pp"
            for phase, values in accuracy_summary.items()
        )
        passed = all(value["within_tolerance"] for value in accuracy_summary.values())
        print(f"  accuracy check {'PASS' if passed else 'FAIL'}: {check_text}", flush=True)
        results.append(
            {
                "name": entry["name"],
                "directory": str(run_directory.relative_to(REPO_ROOT)),
                "model_type": model_type,
                "component_definition": component_definition,
                "seeds": seed_results,
                "traces": traces,
                "accuracy_summary": accuracy_summary,
                "accuracy_tolerance": args.accuracy_tolerance,
            }
        )

    if not results:
        raise RuntimeError("No models selected")

    raw_path = output_dir / "plastic_ratio_raw.csv"
    iteration_path = output_dir / "plastic_ratio_by_iteration.csv"
    phase_path = output_dir / "plastic_ratio_by_phase.csv"
    accuracy_path = output_dir / "accuracy_sanity_check.csv"
    figure_stem = output_dir / "plastic_ratio_by_iteration"
    write_raw_csv(results, reference_tick_phases, raw_path)
    write_iteration_summary_csv(results, reference_tick_phases, iteration_path)
    write_phase_summary_csv(results, reference_tick_phases, phase_path)
    write_accuracy_csv(results, accuracy_path)
    plot_csv(iteration_path, figure_stem)

    all_accuracy_checks_passed = all(
        values["within_tolerance"]
        for result in results
        for values in result["accuracy_summary"].values()
    )
    payload = {
        "protocol": {
            "device": "cpu",
            "candidate_seeds": list(candidate_seeds),
            "successful_seeds_per_model": args.successful_seeds_per_model,
            "success_threshold": args.success_threshold,
            "selected_seeds_by_model": {
                result["name"]: [seed["seed"] for seed in result["seeds"]]
                for result in results
            },
            "samples_per_seed": args.samples_per_seed,
            "evaluation_seed": args.evaluation_seed,
            "ratio": "Frobenius(P^t) / Frobenius(W)",
            "timing": (
                "P^t is read from the post-update recurrent state immediately "
                "after iteration t and is available to iteration t+1"
            ),
            "aggregation": (
                "mean and population standard deviation over checkpoint seeds "
                "and sampled test episodes"
            ),
            "state_reset": "all recurrent and plastic state is reset at each episode",
            "accuracy_tolerance": args.accuracy_tolerance,
            "all_accuracy_checks_passed": all_accuracy_checks_passed,
        },
        "phase_spans": [
            {"phase": phase, "first_iteration": start, "last_iteration": end}
            for phase, start, end in reference_phase_spans
        ],
        "models": serializable_results(results),
    }
    (output_dir / "plastic_ratio_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(f"Wrote results to {output_dir}", flush=True)

    if args.strict_accuracy and not all_accuracy_checks_passed:
        raise SystemExit("At least one aggregate accuracy sanity check failed")


if __name__ == "__main__":
    main()
