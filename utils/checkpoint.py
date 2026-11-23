import glob
import logging
import os
import torch
from typing import Optional

log = logging.getLogger(__name__)


def save_checkpoint(
    epoch: int,
    model_state: dict,
    optimizer_state: dict,
    scheduler_state: dict,
    metrics: dict,
    rng_state: dict,
    checkpoint_dir: str = "."
) -> str:
    model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "metrics": metrics,
        "rng_state": rng_state,
    }

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path: str) -> Optional[dict]:
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint


def get_rng_state() -> dict:
    return {
        "python": torch.get_rng_state(),
        "numpy": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(rng_state: dict):
    torch.set_rng_state(rng_state["python"])
    torch.random.set_rng_state(rng_state["numpy"])
    if rng_state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def find_latest_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        checkpoints = glob.glob(os.path.join(path, "checkpoint_epoch_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in directory: {path}")
        latest = max(checkpoints, key=lambda p: int(p.split("_")[-1].replace(".pt", "")))
        log.info(f"Found latest checkpoint: {latest}")
        return latest
    raise FileNotFoundError(f"Invalid checkpoint path: {path}")
