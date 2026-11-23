"""Analysis helpers for Sequential Split-MNIST."""

from pathlib import Path


def infer_model_type(directory: Path) -> str:
    """Read the model name from current or pre-release experiment directories."""
    parts = directory.parts
    for task_name in ("SequentialSplitMNIST", "ContinualLearningACL"):
        if task_name in parts:
            index = parts.index(task_name) + 1
            if index < len(parts):
                return parts[index]
    raise ValueError(f"Cannot infer the Sequential Split-MNIST model from {directory}")
