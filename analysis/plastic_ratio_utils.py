"""Shared extraction of additive plastic and fixed recurrent components."""

from __future__ import annotations

import torch


def expand_state(state: tuple[torch.Tensor, ...], batch_size: int):
    return tuple(
        value.expand(batch_size, *value.shape[1:]).contiguous()
        if value is not None and value.shape[0] == 1 and batch_size > 1
        else value
        for value in state
    )


def frobenius_norm_per_sample(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(matrix.reshape(matrix.shape[0], -1), dim=1)


def expanded_fixed_norm(weight: torch.Tensor, batch_size: int) -> torch.Tensor:
    return torch.linalg.vector_norm(weight).expand(batch_size)


def plastic_component_norms(
    model_type: str,
    cell,
    state: tuple[torch.Tensor, ...],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return per-sample ||P^t||, ||W||, and the exact decomposition used."""
    if model_type == "GDN":
        # GDN's recurrent state is itself a standalone fast-weight matrix; it
        # is not an additive component paired with a fixed recurrent matrix.
        # A unit reference makes the architecture-specific rho exactly the
        # Frobenius magnitude of that state.
        fast_weight_norm = frobenius_norm_per_sample(state[0])
        reference_norm = fast_weight_norm.new_ones(batch_size)
        definition = "rho^t=||S^t||_F (GDN standalone fast-weight state)"
        return fast_weight_norm, reference_norm, definition
    if model_type == "CTMHebbian":
        fixed = cell.synapses.weights_fixed
        plastic = cell.synapses.alpha.unsqueeze(0) * state[5]
        definition = "W=synapses.weights_fixed; P^t=synapses.alpha*Hebb^t"
    elif model_type in {"CTM", "CTMNoNLMs"} and cell.plastic:
        fixed = cell.synapses.weights_fixed
        state_index = 3 if model_type == "CTMNoNLMs" else 4
        trace = state[state_index].reshape(batch_size, cell.d_model, cell.d_model)
        if cell.synapses.adaptive_scaling:
            trace_norm = frobenius_norm_per_sample(trace).reshape(-1, 1, 1)
            fixed_norm = torch.linalg.vector_norm(fixed)
            scale = fixed_norm * cell.synapses.beta / (trace_norm + 1e-6)
            definition = (
                "W=synapses.weights_fixed; P^t=(||W||_F*beta/"
                "(||S^t||_F+1e-6))*S^t"
            )
        else:
            scale = cell.synapses.fixed_scale_factor
            definition = "W=synapses.weights_fixed; P^t=fixed_scale_factor*S^t"
        plastic = scale * trace
    elif model_type == "PlasticLSTM":
        fixed = cell.weight_h
        # The first three gate blocks are zero in P^t. Zero-padding does not
        # change the norm, so only the plastic candidate-cell block is needed.
        plastic = cell.alpha.unsqueeze(0) * state[2]
        definition = "W=full four-gate weight_h; P^t=[0,0,0,alpha*Hebb^t]"
    elif model_type == "PlasticRNN":
        fixed = cell.weight_h
        plastic = cell.alpha.unsqueeze(0) * state[1]
        definition = "W=weight_h; P^t=alpha*Hebb^t"
    elif model_type == "FW":
        fixed = cell.weight_h
        plastic = state[1]
        definition = "W=weight_h; P^t=A^t (fast-weight state)"
    elif model_type == "STPN":
        fixed = cell.weight
        plastic = state[1]
        definition = "W=weight; P^t=F^t (short-term component)"
    else:
        if model_type == "LSTM":
            fixed = cell.weight_h
        elif model_type == "RNN":
            fixed = cell.weight_h
        elif model_type == "HyperLSTM":
            fixed = cell.weight
        elif model_type == "CTM":
            fixed = cell.synapses.weights_fixed
        else:
            raise ValueError(f"No P/W decomposition is defined for {model_type}")
        plastic = fixed.new_zeros((batch_size, 1))
        if model_type == "HyperLSTM":
            definition = (
                "P^t=0 under the additive decomposition; HyperLSTM's separate "
                "multiplicative modulation is not counted"
            )
        else:
            definition = "P^t=0 (non-plastic architecture)"

    plastic_norm = frobenius_norm_per_sample(plastic)
    fixed_norm = expanded_fixed_norm(fixed, batch_size)
    return plastic_norm, fixed_norm, definition
