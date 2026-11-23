"""Measure architecture-specific rho trajectories on FewShotCIFAR, on CPU.

The script evaluates the runs listed under ``fewshot_cifar`` in
``experiment_dirs.yaml``. It records the scaled post-update additive plastic
component (or the fast-weight-state magnitude for GDN), validates first-query
and all-query accuracy against each run's saved test summary, and invokes
``plot_plastic_ratio.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.fewshot.plot_plastic_ratio import plot_csv  # noqa: E402
from analysis.plastic_ratio_utils import (  # noqa: E402
    expand_state,
    plastic_component_norms,
)
from tasks.few_shot_image_classification import (  # noqa: E402
    DATASET_CONFIGS,
    FewShotDataloader,
    FewShotDataset,
    FewShotImageClassificationModel,
)


DEFAULT_SEEDS = (0, 1, 2, 3, 4)
ENTRY_METADATA = {
    "PlasticCTM": ("Plastic CTM", "CTM"),
    "PlasticLSTM": ("Plastic LSTM (No Mod.)", "PlasticLSTM"),
    "PlasticLSTMSimple": ("Plastic LSTM (Simple Mod.)", "PlasticLSTM"),
    "PlasticLSTMRetro": ("Plastic LSTM (Retro. Mod.)", "PlasticLSTM"),
    "CTM": ("CTM", "CTM"),
    "LSTM": ("LSTM", "LSTM"),
    "RNN": ("RNN", "RNN"),
    "PlasticRNN": ("Plastic RNN", "PlasticRNN"),
    "PlasticRNNOja": ("Plastic RNN (Oja)", "PlasticRNN"),
    "FW": ("FW", "FW"),
    "GDN": ("GDN", "GDN"),
    "HyperLSTM": ("HyperLSTM", "HyperLSTM"),
    "STPN": ("STPN", "STPN"),
    "CTMNoNLMs": ("Plastic CTM (No NLMs)", "CTMNoNLMs"),
    "CTMHebbian": ("Plastic CTM (Hebbian)", "CTMHebbian"),
}
ENTRY_ORDER = tuple(ENTRY_METADATA)


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
        default=Path("outputs/analysis/fewshot/plastic_ratio_cifar"),
    )
    parser.add_argument("--samples-per-seed", type=int, default=64)
    parser.add_argument("--evaluation-seed", type=int, default=12345)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.15)
    parser.add_argument("--strict-accuracy", action="store_true")
    parser.add_argument("--model", action="append")
    parser.add_argument("--seed", action="append", type=int)
    args = parser.parse_args()
    if args.samples_per_seed < 1:
        parser.error("--samples-per-seed must be positive")
    return args


def find_checkpoint(seed_directory: Path) -> Path:
    checkpoints = list(seed_directory.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {seed_directory}")
    return max(checkpoints, key=lambda path: int(path.stem.rsplit("_", 1)[1]))


def compatible_model_config(model_config):
    values = OmegaConf.to_container(model_config, resolve=True)
    aliases = {
        "adaptive_scaling": "plastic_adaptive_scaling",
        "fixed_scale_factor": "plastic_fixed_scale_factor",
    }
    for old_name, new_name in aliases.items():
        if old_name in values and new_name not in values:
            values[new_name] = values.pop(old_name)
    return OmegaConf.create(values)


def normalized_state_dict(checkpoint):
    state_dict = dict(checkpoint["model_state_dict"])
    if state_dict and all(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
        }
    old_suffix = "synapses.parametrizations.target_plastic_ratio.original"
    new_suffix = "synapses.parametrizations.beta.original"
    for key in list(state_dict):
        if key.endswith(old_suffix):
            state_dict[key[: -len(old_suffix)] + new_suffix] = state_dict.pop(key)
    return state_dict


def load_model(
    seed_directory: Path,
    model_type: str,
    dataset_name: str | None = None,
    device: str | torch.device = "cpu",
):
    config = OmegaConf.load(seed_directory / ".hydra" / "config.yaml")
    checkpoint_path = find_checkpoint(seed_directory)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = normalized_state_dict(checkpoint)
    saved_model_values = OmegaConf.to_container(config.model, resolve=True)
    config.model = compatible_model_config(config.model)
    has_saved_scaling_setting = any(
        key in saved_model_values
        for key in (
            "adaptive_scaling",
            "fixed_scale_factor",
            "plastic_adaptive_scaling",
            "plastic_fixed_scale_factor",
        )
    )
    has_saved_beta = any(
        key.endswith("synapses.parametrizations.beta.original")
        for key in state_dict
    )
    # Legacy-only fallback: this old ablation predates adaptive plastic
    # scaling. New checkpoints carry either an explicit setting or beta and
    # therefore bypass this branch.
    if model_type == "CTMNoNLMs" and not has_saved_scaling_setting and not has_saved_beta:
        config.model.plastic_adaptive_scaling = False
        config.model.plastic_fixed_scale_factor = 1.0
    if model_type == "CTMHebbian" and "rule" in config.model:
        if config.model.rule != "hebb":
            raise ValueError(
                f"This CTMHebbian implementation cannot reproduce rule={config.model.rule!r}"
            )
        del config.model.rule
    config.model.d_model = int(state_dict["output_proj.weight"].shape[1])
    device = torch.device(device)
    model = FewShotImageClassificationModel(
        model_cfg=config.model,
        task_cfg=config.task,
        model_type=model_type,
    ).to(device)
    configured_dataset = config.task.get("dataset")
    if dataset_name is not None:
        if configured_dataset is not None and configured_dataset != dataset_name:
            raise ValueError(
                f"Checkpoint config uses {configured_dataset}, not {dataset_name}"
            )
    elif configured_dataset is not None:
        dataset_name = configured_dataset
    else:
        projection_width = int(state_dict["backbone.output_proj.weight"].shape[1])
        n_classes = int(config.task.nKnovel)
        matches = [
            name
            for name, dataset_config in DATASET_CONFIGS.items()
            if 64 * (int(dataset_config["image_size"]) // 16) ** 2 + n_classes
            == projection_width
        ]
        if len(matches) != 1:
            raise ValueError(
                "Cannot infer legacy checkpoint dataset from backbone projection "
                f"width {projection_width}"
            )
        dataset_name = matches[0]
    config.task.dataset = dataset_name
    image_size = int(DATASET_CONFIGS[dataset_name]["image_size"])
    with torch.inference_mode():
        model(
            torch.zeros(1, 1, 3, image_size, image_size, device=device),
            torch.zeros(1, 1, int(config.task.nKnovel), device=device),
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    # Historical non-plastic CTM checkpoints predate registration of beta on
    # PlasticLinear. The current constructor creates beta even when plastic is
    # false, but it is never read by the non-plastic forward path.
    allowed_missing = set()
    if model_type == "CTM" and not bool(config.model.plastic):
        allowed_missing = {
            key for key in missing
            if key.endswith("synapses.parametrizations.beta.original")
        }
    if missing - allowed_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint incompatibility: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.eval()
    return model, config, checkpoint_path


def batch_key(config, sample_count: int, evaluation_seed: int):
    return (
        config.task.get("dataset", "CIFAR100"),
        int(config.task.iterations),
        int(config.task.nKnovel),
        int(config.task.nKbase),
        int(config.task.nExemplars),
        int(config.task.nTestNovel),
        int(config.task.nTestBase),
        sample_count,
        evaluation_seed,
    )


def sample_test_batch(config, sample_count: int, evaluation_seed: int):
    random.seed(evaluation_seed)
    np.random.seed(evaluation_seed)
    torch.manual_seed(evaluation_seed)
    dataset = FewShotDataset(
        dataset_name=config.task.get("dataset", "CIFAR100"), phase="test"
    )
    loader = FewShotDataloader(
        dataset=dataset,
        internal_ticks=config.task.iterations,
        nKnovel=config.task.nKnovel,
        nKbase=config.task.nKbase,
        nExemplars=config.task.nExemplars,
        nTestNovel=config.task.nTestNovel,
        nTestBase=config.task.nTestBase,
        batch_size=sample_count,
        epoch_size=sample_count,
        num_workers=0,
    )
    episodes = [loader.load_function(index) for index in range(sample_count)]
    inputs = torch.stack([episode[0][0] for episode in episodes], dim=0)
    auxiliary = torch.stack([episode[0][1] for episode in episodes], dim=0)
    targets = torch.stack([episode[1] for episode in episodes], dim=0)
    return (
        inputs.transpose(0, 1).contiguous(),
        auxiliary.transpose(0, 1).contiguous(),
        targets.transpose(0, 1).contiguous(),
    )


def phase_layout(config):
    ticks = int(config.task.iterations)
    support = int(config.task.nKnovel) * int(config.task.nExemplars) * ticks
    queries = (
        int(config.task.nTestNovel) + int(config.task.nTestBase)
    ) * ticks
    phases = ["Train"] * support + ["Test"] * queries
    spans = [("Train", 1, support), ("Test", support + 1, support + queries)]
    return phases, spans


def evaluate_trajectories(model, model_type: str, config, batch):
    inputs, auxiliary, targets = batch
    time_steps, batch_size, channels, height, width = inputs.shape
    ticks = int(config.task.iterations)
    support_ticks = (
        int(config.task.nKnovel) * int(config.task.nExemplars) * ticks
    )
    query_count = int(config.task.nTestNovel) + int(config.task.nTestBase)
    query_indices = [support_ticks + (index + 1) * ticks - 1 for index in range(query_count)]
    first_query_index = query_indices[0]

    ratios, plastic_norms, fixed_norms = [], [], []
    counts = {
        "first_query": {"correct": 0, "total": 0},
        "all_queries": {"correct": 0, "total": 0},
    }
    definition = None
    cell = model.rnn.recurrent_module
    with torch.inference_mode():
        embeddings = model.backbone(
            inputs.reshape(time_steps * batch_size, channels, height, width),
            auxiliary.reshape(time_steps * batch_size, -1),
        ).view(time_steps, batch_size, -1)
        state = expand_state(model.rnn.get_initial_state(), batch_size)
        for iteration in range(time_steps):
            hidden, state = cell(embeddings[iteration], state)
            p_norm, w_norm, current_definition = plastic_component_norms(
                model_type, cell, state, batch_size
            )
            if definition is None:
                definition = current_definition
            elif definition != current_definition:
                raise RuntimeError("P/W definition changed during the episode")
            ratios.append(p_norm / w_norm.clamp_min(torch.finfo(w_norm.dtype).eps))
            plastic_norms.append(p_norm)
            fixed_norms.append(w_norm)

            if iteration in query_indices:
                predictions = model.output_proj(hidden).argmax(dim=-1)
                correct = int((predictions == targets[iteration]).sum().item())
                counts["all_queries"]["correct"] += correct
                counts["all_queries"]["total"] += batch_size
                if iteration == first_query_index:
                    counts["first_query"]["correct"] += correct
                    counts["first_query"]["total"] += batch_size

    return {
        "ratio": torch.stack(ratios, dim=1).cpu().numpy(),
        "plastic_norm": torch.stack(plastic_norms, dim=1).cpu().numpy(),
        "fixed_norm": torch.stack(fixed_norms, dim=1).cpu().numpy(),
        "accuracy_counts": counts,
        "component_definition": definition,
    }


def saved_accuracy(seed_directory: Path):
    with (seed_directory / "summary.json").open() as file:
        metrics = json.load(file)["test_metrics"]
    return {
        "first_query": float(metrics["accuracy"]),
        "all_queries": float(metrics["accuracy_all_images"]),
    }


def load_entries(config_path: Path, selected_models: set[str]):
    with config_path.open() as file:
        section = yaml.safe_load(file)["fewshot_cifar"]
    entries = []
    for key in ENTRY_ORDER:
        if key not in section:
            continue
        name, model_type = ENTRY_METADATA[key]
        if selected_models and name not in selected_models:
            continue
        directories = section[key]
        if isinstance(directories, str):
            directories = [directories]
        for directory in directories:
            entries.append(
                {"key": key, "name": name, "model_type": model_type, "directory": directory}
            )
    if selected_models:
        found = {entry["name"] for entry in entries}
        unknown = selected_models - found
        if unknown:
            raise ValueError(f"Unknown model name(s): {', '.join(sorted(unknown))}")
    return entries


def accuracy_summary(seed_results, tolerance: float):
    summary = {}
    for metric in ("first_query", "all_queries"):
        correct = sum(result["accuracy_counts"][metric]["correct"] for result in seed_results)
        total = sum(result["accuracy_counts"][metric]["total"] for result in seed_results)
        sampled = correct / total
        saved = [result["saved_accuracy"][metric] for result in seed_results]
        difference = abs(sampled - float(np.mean(saved)))
        summary[metric] = {
            "sampled": sampled,
            "sample_correct": correct,
            "sample_total": total,
            "saved_mean": float(np.mean(saved)),
            "saved_std_across_seeds": float(np.std(saved)),
            "absolute_difference": difference,
            "within_tolerance": difference <= tolerance,
        }
    return summary


def write_raw_csv(results, phases, path: Path):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Model", "Seed", "Sample", "Iteration", "Phase", "Plastic norm", "Fixed norm", "Plastic/fixed ratio"]
        )
        for result in results:
            for seed in result["seeds"]:
                for sample in range(seed["ratio"].shape[0]):
                    for index, phase in enumerate(phases):
                        writer.writerow(
                            [
                                result["name"], seed["seed"], sample, index + 1, phase,
                                seed["plastic_norm"][sample, index], seed["fixed_norm"][sample, index],
                                seed["ratio"][sample, index],
                            ]
                        )


def write_iteration_csv(results, phases, path: Path):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Iteration", "Phase", "Mean ratio", "Standard deviation", "N"])
        for result in results:
            for index, phase in enumerate(phases):
                values = result["traces"][:, index]
                writer.writerow([result["name"], index + 1, phase, values.mean(), values.std(), len(values)])


def write_phase_csv(results, phases, path: Path):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Phase", "Mean ratio", "Standard deviation", "N"])
        for result in results:
            for phase in dict.fromkeys(phases):
                indices = [index for index, value in enumerate(phases) if value == phase]
                values = result["traces"][:, indices].reshape(-1)
                writer.writerow([result["name"], phase, values.mean(), values.std(), len(values)])


def write_accuracy_csv(results, path: Path):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Model", "Metric", "Sampled accuracy", "Sample correct", "Sample total", "Saved accuracy mean", "Saved accuracy std", "Absolute difference", "Within tolerance"]
        )
        for result in results:
            for metric, values in result["accuracy_summary"].items():
                writer.writerow(
                    [
                        result["name"], metric, values["sampled"], values["sample_correct"],
                        values["sample_total"], values["saved_mean"], values["saved_std_across_seeds"],
                        values["absolute_difference"], values["within_tolerance"],
                    ]
                )


def serializable_results(results):
    return [
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
                }
                for seed in result["seeds"]
            ],
        }
        for result in results
    ]


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    seeds = tuple(dict.fromkeys(args.seed or DEFAULT_SEEDS))
    entries = load_entries(args.config.resolve(), set(args.model or []))
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    batches, results = {}, []
    reference_phases = reference_spans = None
    for entry in entries:
        print(f"{entry['name']} ({entry['model_type']})", flush=True)
        run_directory = REPO_ROOT / entry["directory"]
        seed_results = []
        definition = None
        for seed in seeds:
            seed_directory = run_directory / str(seed)
            model, config, checkpoint_path = load_model(seed_directory, entry["model_type"])
            key = batch_key(config, args.samples_per_seed, args.evaluation_seed)
            if key not in batches:
                batches[key] = sample_test_batch(config, args.samples_per_seed, args.evaluation_seed)
            phases, spans = phase_layout(config)
            if reference_phases is None:
                reference_phases, reference_spans = phases, spans
            elif reference_phases != phases:
                raise ValueError("Selected runs do not share an episode layout")
            evaluated = evaluate_trajectories(model, entry["model_type"], config, batches[key])
            if definition is None:
                definition = evaluated["component_definition"]
            elif definition != evaluated["component_definition"]:
                raise RuntimeError("P/W definition differs across seeds")
            evaluated.update(
                {
                    "seed": seed,
                    "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
                    "saved_accuracy": saved_accuracy(seed_directory),
                }
            )
            seed_results.append(evaluated)
            print(
                "  seed " + str(seed) + ": " + ", ".join(
                    f"{name}={100 * value['correct'] / value['total']:.1f}%"
                    for name, value in evaluated["accuracy_counts"].items()
                ),
                flush=True,
            )
            del model
        summary = accuracy_summary(seed_results, args.accuracy_tolerance)
        passed = all(value["within_tolerance"] for value in summary.values())
        print(
            f"  accuracy check {'PASS' if passed else 'FAIL'}: "
            + ", ".join(
                f"{name} Δ={100 * value['absolute_difference']:.1f}pp"
                for name, value in summary.items()
            ),
            flush=True,
        )
        results.append(
            {
                "name": entry["name"],
                "directory": entry["directory"],
                "model_type": entry["model_type"],
                "component_definition": definition,
                "seeds": seed_results,
                "traces": np.concatenate([seed["ratio"] for seed in seed_results], axis=0),
                "accuracy_summary": summary,
            }
        )

    raw_path = output_dir / "plastic_ratio_raw.csv"
    iteration_path = output_dir / "plastic_ratio_by_iteration.csv"
    write_raw_csv(results, reference_phases, raw_path)
    write_iteration_csv(results, reference_phases, iteration_path)
    write_phase_csv(results, reference_phases, output_dir / "plastic_ratio_by_phase.csv")
    write_accuracy_csv(results, output_dir / "accuracy_sanity_check.csv")
    plot_csv(iteration_path, output_dir / "plastic_ratio_by_iteration")

    all_passed = all(
        value["within_tolerance"]
        for result in results
        for value in result["accuracy_summary"].values()
    )
    payload = {
        "protocol": {
            "device": "cpu",
            "dataset": "CIFAR100",
            "seeds": list(seeds),
            "samples_per_seed": args.samples_per_seed,
            "evaluation_seed": args.evaluation_seed,
            "rho": (
                "Frobenius(P^t) / Frobenius(W) for additive-plastic and "
                "non-plastic architectures; Frobenius(S^t) for GDN"
            ),
            "timing": "post-update P^t, available to iteration t+1",
            "aggregation": "population mean/std over checkpoint seeds and test episodes",
            "state_reset": "at every episode",
            "accuracy_tolerance": args.accuracy_tolerance,
            "all_accuracy_checks_passed": all_passed,
        },
        "phase_spans": [
            {"phase": phase, "first_iteration": start, "last_iteration": end}
            for phase, start, end in reference_spans
        ],
        "models": serializable_results(results),
    }
    (output_dir / "plastic_ratio_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote results to {output_dir}", flush=True)
    if args.strict_accuracy and not all_passed:
        raise SystemExit("At least one accuracy sanity check failed")


if __name__ == "__main__":
    main()
