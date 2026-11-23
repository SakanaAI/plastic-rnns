"""Shared helpers for paired inference-time plasticity ablations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from analysis.plastic_ratio_utils import expand_state


SUPPORTED_MODEL_TYPES = {
    "PlasticRNN",
    "PlasticLSTM",
    "FW",
    "STPN",
    "CTM",
    "CTMNoNLMs",
    "CTMHebbian",
}


def disable_plasticity(model, model_type: str) -> str:
    """Remove the trained model's additive plastic component exactly.

    Plastic traces may continue to evolve for models where they are separate
    from their learned contribution coefficient. The intervention guarantees
    that those traces do not contribute to the effective recurrent weights.
    """
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"No plasticity-disabling intervention for {model_type}")

    cell = model.rnn.recurrent_module
    with torch.no_grad():
        if model_type in {"PlasticRNN", "PlasticLSTM"}:
            cell.alpha.zero_()
            description = "alpha=0"
            disabled_value = cell.alpha
        elif model_type == "STPN":
            cell.Gamma.zero_()
            cell.F_0.zero_()
            description = "Gamma=0 with F^0=0"
            disabled_value = cell.Gamma
        elif model_type == "FW":
            cell.eta.zero_()
            cell.A_0.zero_()
            description = "eta=0 with A^0=0"
            disabled_value = cell.eta
        elif model_type == "CTMHebbian":
            cell.synapses.alpha.zero_()
            description = "synaptic alpha=0"
            disabled_value = cell.synapses.alpha
        else:
            if not bool(cell.plastic):
                raise ValueError(f"{model_type} checkpoint is not plastic")
            # Switching to the fixed-scale branch avoids the bounded beta
            # parametrization, whose inverse cannot represent exactly zero.
            cell.synapses.adaptive_scaling = False
            cell.synapses.fixed_scale_factor = 0.0
            description = "plastic scale=0"
            disabled_value = cell.synapses.weights_fixed.new_zeros(())

    if torch.count_nonzero(disabled_value).item() != 0:
        raise RuntimeError(f"Plasticity intervention failed for {model_type}")
    return description


def encode_fewshot(model, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a materialized few-shot batch once for a paired comparison."""
    inputs, auxiliary, targets = batch
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    auxiliary = auxiliary.to(device)
    targets = targets.to(device)
    time_steps, batch_size, channels, height, width = inputs.shape
    with torch.inference_mode():
        embeddings = model.backbone(
            inputs.reshape(time_steps * batch_size, channels, height, width),
            auxiliary.reshape(time_steps * batch_size, -1),
        ).view(time_steps, batch_size, -1)
    return embeddings, targets


def encode_continual(model, batch) -> tuple[torch.Tensor, torch.Tensor, Mapping]:
    """Encode a materialized continual-learning batch once."""
    inputs, auxiliary, targets, sequence_info = batch
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    auxiliary = auxiliary.to(device)
    targets = targets.to(device)
    time_steps, batch_size, channels, height, width = inputs.shape
    with torch.inference_mode():
        flat_auxiliary = auxiliary.reshape(time_steps * batch_size)
        label_embeddings = torch.nn.functional.one_hot(
            flat_auxiliary, num_classes=model.num_classes + 1
        ).float()
        embeddings = model.backbone(
            inputs.reshape(time_steps * batch_size, channels, height, width),
            label_embeddings,
        ).view(time_steps, batch_size, -1)
    return embeddings, targets, sequence_info


def evaluate_encoded(
    model,
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    metric_indices: Mapping[str, Sequence[int]],
) -> dict[str, dict[str, int]]:
    """Evaluate query accuracies from a fresh recurrent state."""
    batch_size = embeddings.shape[1]
    indices_to_metrics: dict[int, list[str]] = {}
    for metric, indices in metric_indices.items():
        for index in indices:
            indices_to_metrics.setdefault(int(index), []).append(metric)

    counts = {
        metric: {"correct": 0, "total": 0}
        for metric in metric_indices
    }
    cell = model.rnn.recurrent_module
    state = expand_state(model.rnn.get_initial_state(), batch_size)
    with torch.inference_mode():
        for iteration in range(embeddings.shape[0]):
            hidden, state = cell(embeddings[iteration], state)
            if iteration not in indices_to_metrics:
                continue
            predictions = model.output_proj(hidden).argmax(dim=-1)
            correct = int((predictions == targets[iteration]).sum().item())
            for metric in indices_to_metrics[iteration]:
                counts[metric]["correct"] += correct
                counts[metric]["total"] += batch_size
    return counts


def paired_evaluation(
    model,
    model_type: str,
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    metric_indices: Mapping[str, Sequence[int]],
) -> tuple[dict, dict, str]:
    """Evaluate enabled and disabled conditions on the exact same episodes."""
    enabled = evaluate_encoded(model, embeddings, targets, metric_indices)
    intervention = disable_plasticity(model, model_type)
    disabled = evaluate_encoded(model, embeddings, targets, metric_indices)
    return enabled, disabled, intervention


def summarize_paired(seed_results: list[dict], metrics: Sequence[str]) -> dict:
    """Pool counts across seeds and report absolute and relative changes."""
    summary = {}
    for metric in metrics:
        enabled_correct = sum(
            result["enabled"][metric]["correct"] for result in seed_results
        )
        enabled_total = sum(
            result["enabled"][metric]["total"] for result in seed_results
        )
        disabled_correct = sum(
            result["disabled"][metric]["correct"] for result in seed_results
        )
        disabled_total = sum(
            result["disabled"][metric]["total"] for result in seed_results
        )
        enabled_accuracy = enabled_correct / enabled_total
        disabled_accuracy = disabled_correct / disabled_total
        absolute_change = disabled_accuracy - enabled_accuracy
        relative_change = (
            100.0 * absolute_change / enabled_accuracy
            if enabled_accuracy > 0
            else None
        )
        seed_relative_changes = []
        for result in seed_results:
            seed_enabled = (
                result["enabled"][metric]["correct"]
                / result["enabled"][metric]["total"]
            )
            seed_disabled = (
                result["disabled"][metric]["correct"]
                / result["disabled"][metric]["total"]
            )
            seed_relative_changes.append(
                100.0 * (seed_disabled - seed_enabled) / seed_enabled
                if seed_enabled > 0
                else None
            )
        valid_seed_changes = [
            value for value in seed_relative_changes if value is not None
        ]
        summary[metric] = {
            "enabled_accuracy": enabled_accuracy,
            "disabled_accuracy": disabled_accuracy,
            "absolute_change_percentage_points": 100.0 * absolute_change,
            "relative_change_percent": relative_change,
            "enabled_correct": enabled_correct,
            "enabled_total": enabled_total,
            "disabled_correct": disabled_correct,
            "disabled_total": disabled_total,
            "seed_relative_changes_percent": seed_relative_changes,
            "mean_seed_relative_change_percent": (
                float(torch.tensor(valid_seed_changes).mean())
                if valid_seed_changes else None
            ),
            "std_seed_relative_change_percent": (
                float(torch.tensor(valid_seed_changes).std(correction=0))
                if valid_seed_changes else None
            ),
        }
    return summary
