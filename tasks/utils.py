import inspect
import logging

import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def get_device(cfg):
    device_id = cfg.device
    if device_id != -1:
        device = f'cuda:{device_id}'
    else:
        device = 'cpu'
    return device


def _build_and_count(task_cls, cfg: DictConfig, model_type: str) -> int:
    """Build model on CPU, materialize lazy modules, and return trainable parameter count."""
    model = task_cls.build_model(cfg, model_type)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    return count


def optimise_model_size(task_cls, cfg: DictConfig, model_type: str) -> DictConfig:
    """Fit a model's scalable width to ``cfg.max_parameters``.

    Recurrent models scale ``d_model``; the Conv4 MAML and ProtoNet baselines
    scale their channel width. Keeping the width out of the Optuna search makes
    capacity an experimental control rather than a tuned hyperparameter.
    """
    # Check that the task supports build_model as a static/classmethod
    raw = inspect.getattr_static(task_cls, 'build_model')
    if not isinstance(raw, (staticmethod, classmethod)):
        log.warning(f"max_parameters not supported for {task_cls.__name__} (no static build_model)")
        return cfg

    if model_type in {"MAML", "ProtoNet"}:
        size_parameter = "hidden_channels"
        low, high = 4, 1024
    else:
        size_parameter = "d_model"
        low, high = 8, 4096

    best = low
    best_count = None
    while low <= high:
        mid = (low + high) // 2
        cfg.model[size_parameter] = mid
        count = _build_and_count(task_cls, cfg, model_type)
        if count <= cfg.max_parameters:
            best = mid
            best_count = count
            low = mid + 1
        else:
            high = mid - 1

    cfg.model[size_parameter] = best
    if best_count is None:
        minimum_count = _build_and_count(task_cls, cfg, model_type)
        raise ValueError(
            f"Even {size_parameter}={best} gives {minimum_count} parameters, "
            f"above the budget of {cfg.max_parameters}"
        )
    log.info(
        "Optimised %s=%d (%d parameters; budget=%d)",
        size_parameter,
        best,
        best_count,
        cfg.max_parameters,
    )
    return cfg


# Backwards-compatible import for code outside this repository.
optimise_d_model = optimise_model_size


def get_target_plastic_ratio(model):
    """
    Extract target_plastic_ratio parameter value from a model containing CTM cells.
    Returns None if the model doesn't have this parameter.
    """
    for module in model.modules():
        if hasattr(module, 'get_target_plastic_ratio'):
            return module.get_target_plastic_ratio()
    return None

def get_plastic_contribution_ratio(model):
    """
    Extract the achieved plastic/fixed ratio from a model containing CTM cells.
    Returns None if the model doesn't have this metric.
    """
    for module in model.modules():
        if hasattr(module, 'get_plastic_contribution_ratio'):
            return module.get_plastic_contribution_ratio()
    return None

def get_raw_plastic_ratio(model):
    """
    Extract the raw plastic/fixed ratio (before adaptive scaling) from a model containing CTM cells.
    Returns None if the model doesn't have this metric.
    """
    for module in model.modules():
        if hasattr(module, 'get_raw_plastic_ratio'):
            return module.get_raw_plastic_ratio()
    return None

def get_scale_factor(model):
    """
    Extract the adaptive scale factor applied to plastic weights from a model containing CTM cells.
    Returns None if the model doesn't have this metric.
    """
    for module in model.modules():
        if hasattr(module, 'get_scale_factor'):
            return module.get_scale_factor()
    return None
