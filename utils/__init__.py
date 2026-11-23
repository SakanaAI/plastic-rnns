import random

import numpy as np
import torch

from utils.logging import LocalLogger
from utils.plotting import plot_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint, get_rng_state, set_rng_state, find_latest_checkpoint

__all__ = ["LocalLogger", "plot_metrics", "set_random_seeds", "save_checkpoint", "load_checkpoint", "get_rng_state", "set_rng_state", "find_latest_checkpoint"]


def set_random_seeds(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False