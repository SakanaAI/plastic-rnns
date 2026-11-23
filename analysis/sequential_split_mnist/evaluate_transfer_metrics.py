"""Evaluate GEM-style transfer metrics on Sequential Split-MNIST checkpoints.

The trained parameters are fixed during evaluation and the recurrent state is
treated as the within-episode learner state. For the two tasks A and B, matched
query examples are used to calculate BWT = R_BA - R_AA and FWT = R_AB - bar_b_B.
Each query phase is evaluated from a cloned anchor state so that it cannot alter
the state used for a later task. State evolution within each query phase is kept
because it is part of the trained recurrent evaluation protocol.
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

from analysis.sequential_split_mnist import infer_model_type  # noqa: E402
from tasks.sequential_split_mnist import (  # noqa: E402
    SequentialSplitMNISTDataloader,
    SequentialSplitMNISTDataset,
    SequentialSplitMNISTModel,
)


METRICS = (
    "set_b_baseline",
    "accuracy_immediate",
    "accuracy_set_b_before_exemplars",
    "accuracy_retention",
    "accuracy_set_b",
    "fwt",
    "bwt",
)
SEEDS = (0, 1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("experiment_dirs.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/sequential_split_mnist"),
    )
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--evaluation-seed", type=int, default=12345)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Evaluate only the named model. May be supplied more than once.",
    )
    return parser.parse_args()


def find_checkpoint(seed_directory: Path) -> Path:
    checkpoints = list(seed_directory.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {seed_directory}")
    return max(checkpoints, key=lambda path: int(path.stem.rsplit("_", 1)[1]))


def load_model(seed_directory: Path, model_type: str, device: torch.device):
    config = OmegaConf.load(seed_directory / ".hydra" / "config.yaml")
    checkpoint_path = find_checkpoint(seed_directory)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    if state_dict and all(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
        }
    config.model.d_model = int(state_dict["output_proj.weight"].shape[1])

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

    model.load_state_dict(state_dict)
    model.eval()
    return model, config, checkpoint_path


def clone_state(state):
    return tuple(value.clone() if value is not None else None for value in state)


def expand_state(state, batch_size: int):
    return tuple(
        value.expand(batch_size, *value.shape[1:]).contiguous()
        if value is not None and value.shape[0] == 1 and batch_size > 1
        else value
        for value in state
    )


def run_ticks(model, embeddings, state, indices):
    state = clone_state(state)
    for index in indices:
        _, state = model.rnn(embeddings[index : index + 1], state)
    return state


def query_tick_groups(last_tick_indices, internal_ticks: int):
    return [
        list(range(last_tick - internal_ticks + 1, last_tick + 1))
        for last_tick in last_tick_indices
    ]


def evaluate_queries(model, embeddings, targets, anchor_state, query_groups):
    correct = 0
    total = 0
    query_state = clone_state(anchor_state)
    for tick_group in query_groups:
        output = None
        for index in tick_group:
            output, query_state = model.rnn(
                embeddings[index : index + 1], query_state
            )
        logits = model.output_proj(output[0])
        predictions = logits.argmax(dim=-1)
        correct += (predictions == targets[tick_group[-1]]).sum().item()
        total += predictions.numel()
    return correct, total


def evaluate_checkpoint(
    model,
    config,
    device: torch.device,
    max_batches: int,
    evaluation_seed: int,
):
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
        batch_size=config.training.batch_size,
        epoch_size=config.training.batch_size * max_batches,
    )

    counts = {
        name: {"correct": 0, "total": 0}
        for name in (
            "set_b_baseline",
            "accuracy_immediate",
            "accuracy_set_b_before_exemplars",
            "accuracy_retention",
            "accuracy_set_b",
        )
    }

    internal_ticks = dataloader.internal_ticks
    n_exemplars = dataloader.n_exemplars_total
    n_queries = dataloader.queries_per_loss
    a_exemplar_ticks = list(range(n_exemplars * internal_ticks))
    b_exemplar_start = (n_exemplars + n_queries) * internal_ticks
    b_exemplar_ticks = list(
        range(b_exemplar_start, b_exemplar_start + n_exemplars * internal_ticks)
    )
    a_query_groups = query_tick_groups(
        dataloader.sequence_info["immediate_query_indices"], internal_ticks
    )
    b_query_groups = query_tick_groups(
        dataloader.sequence_info["forward_query_indices"], internal_ticks
    )
    retention_query_groups = query_tick_groups(
        dataloader.sequence_info["backward_query_indices"], internal_ticks
    )

    with torch.inference_mode():
        for (inputs, aux_inputs), targets in dataloader:
            inputs = inputs.transpose(0, 1).contiguous().to(device)
            aux_inputs = aux_inputs.transpose(0, 1).contiguous().to(device)
            targets = targets.transpose(0, 1).contiguous().to(device)
            time_steps, batch_size, channels, height, width = inputs.shape

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

            state_0 = expand_state(model.rnn.get_initial_state(), batch_size)
            # State after completing the Set A exemplar phase.
            state_a = run_ticks(model, embeddings, state_0, a_exemplar_ticks)
            # The original episode includes Set A queries before Set B exemplars.
            # Preserve that path, but do not carry state from any added transfer
            # evaluation branch into the episode.
            a_query_ticks = [index for group in a_query_groups for index in group]
            state_a_after_queries = run_ticks(
                model, embeddings, state_a, a_query_ticks
            )
            state_ab = run_ticks(
                model, embeddings, state_a_after_queries, b_exemplar_ticks
            )
            retention_query_ticks = [
                index for group in retention_query_groups for index in group
            ]
            state_ab_after_retention = run_ticks(
                model, embeddings, state_ab, retention_query_ticks
            )

            evaluations = {
                "set_b_baseline": (state_0, b_query_groups),
                "accuracy_immediate": (state_a, a_query_groups),
                "accuracy_set_b_before_exemplars": (
                    state_a_after_queries,
                    b_query_groups,
                ),
                "accuracy_retention": (state_ab, a_query_groups),
                "accuracy_set_b": (state_ab_after_retention, b_query_groups),
            }
            for name, (anchor_state, query_groups) in evaluations.items():
                correct, total = evaluate_queries(
                    model, embeddings, targets, anchor_state, query_groups
                )
                counts[name]["correct"] += correct
                counts[name]["total"] += total

    metrics = {
        name: value["correct"] / value["total"] for name, value in counts.items()
    }
    metrics["fwt"] = (
        metrics["accuracy_set_b_before_exemplars"] - metrics["set_b_baseline"]
    )
    metrics["bwt"] = metrics["accuracy_retention"] - metrics["accuracy_immediate"]
    return metrics


def summarize(seed_results):
    return {
        metric: {
            "mean": float(np.mean([result["metrics"][metric] for result in seed_results])),
            "std": float(np.std([result["metrics"][metric] for result in seed_results])),
        }
        for metric in METRICS
    }


def write_csv(results, output_path: Path):
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Seed", *METRICS])
        for model_result in results:
            for seed_result in model_result["seeds"]:
                writer.writerow(
                    [
                        model_result["name"],
                        seed_result["seed"],
                        *[seed_result["metrics"][metric] for metric in METRICS],
                    ]
                )


def latex_model_name(name: str) -> str:
    return name.replace("&", r"\&")


def format_transfer_value(mean: float, standard_deviation: float) -> str:
    mean_text = f"{abs(mean):.2f}"
    value = (
        f"${mean_text} "
        + r"\scalebox{0.85}{$\pm "
        + f"{standard_deviation:.2f}"
        + r"$}$"
    )
    if mean > 0:
        return r"\textcolor{green}{$\boldsymbol{\uparrow}$}\," + value
    if mean < 0:
        return r"\textcolor{red}{$\boldsymbol{\downarrow}$}\," + value
    return value


def write_latex_table(results, output_path: Path):
    lines = [
        r"\begin{table}[t!]",
        r"\caption{\NEW{Forward transfer and backward transfer on episodic in-context Sequential Split-MNIST. Values are percentage-point differences reported as the mean $\pm$ standard deviation over five seeds. Green upward arrows indicate positive transfer and red downward arrows indicate negative transfer.}}",
        r"\label{tab:splitmnist-transfer}",
        r"\begin{center}",
        r"\begin{small}",
        r"\begin{sc}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Architecture & FWT & BWT \\",
        r"\midrule",
    ]
    for result in results:
        fwt = 100 * result["summary"]["fwt"]["mean"]
        fwt_std = 100 * result["summary"]["fwt"]["std"]
        bwt = 100 * result["summary"]["bwt"]["mean"]
        bwt_std = 100 * result["summary"]["bwt"]["std"]
        lines.append(
            rf"\NEWrow {latex_model_name(result['name'])} & "
            f"{format_transfer_value(fwt, fwt_std)} & "
            f"{format_transfer_value(bwt, bwt_std)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{sc}",
            r"\end{small}",
            r"\end{center}",
            r"\vskip -0.1in",
            r"\end{table}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    with args.config.resolve().open() as file:
        experiment_config = yaml.safe_load(file)

    selected_models = set(args.model or [])
    device = torch.device(args.device)
    results = []
    for model_entry in experiment_config["models"]:
        if selected_models and model_entry["name"] not in selected_models:
            continue
        run_directory = REPO_ROOT / model_entry["directory"]
        model_type = infer_model_type(run_directory)
        seed_results = []
        for seed in SEEDS:
            seed_directory = run_directory / str(seed)
            print(f"Evaluating {model_entry['name']} seed {seed}", flush=True)
            model, config, checkpoint_path = load_model(
                seed_directory, model_type, device
            )
            metrics = evaluate_checkpoint(
                model,
                config,
                device=device,
                max_batches=args.max_batches,
                evaluation_seed=args.evaluation_seed,
            )
            seed_results.append(
                {
                    "seed": seed,
                    "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
                    "metrics": metrics,
                }
            )
            print(
                f"  FWT {100 * metrics['fwt']:+.2f} pp, "
                f"BWT {100 * metrics['bwt']:+.2f} pp",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results.append(
            {
                "name": model_entry["name"],
                "directory": model_entry["directory"],
                "seeds": seed_results,
                "summary": summarize(seed_results),
            }
        )

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "definition": "GEM-style metrics adapted to within-episode recurrent-state learning",
            "fwt": "R_A,B - bar_b_B",
            "bwt": "R_B,A - R_A,A",
            "matrix_entries": {
                "bar_b_B": "set_b_baseline",
                "R_A,A": "accuracy_immediate",
                "R_A,B": "accuracy_set_b_before_exemplars",
                "R_B,A": "accuracy_retention",
                "R_B,B": "accuracy_set_b",
            },
            "query_state_updates_persist_within_phase": True,
            "query_state_updates_affect_later_phases": False,
            "max_batches": args.max_batches,
            "evaluation_seed": args.evaluation_seed,
        },
        "models": results,
    }
    (output_dir / "transfer_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    write_csv(results, output_dir / "transfer_metrics.csv")
    write_latex_table(results, output_dir / "transfer_metrics_table.tex")


if __name__ == "__main__":
    main()
