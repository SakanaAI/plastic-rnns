from abc import abstractmethod
import torch.nn as nn
from torch import Tensor

class RecurrentCell(nn.Module):
    """Base class for recurrent neural network cells."""
    
    def __init__(self, d_input: int, d_model: int):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
    
    @abstractmethod
    def initialize_state(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, input: Tensor, state):
        raise NotImplementedError

    @staticmethod
    def _expand_state(state: tuple, batch_size: int) -> tuple:
        """Expand state tensors from batch size 1 to match the input batch size."""
        if state[0].shape[0] == 1 and batch_size > 1:
            return tuple(s.expand(batch_size, *(-1,) * (s.ndim - 1)) for s in state)
        return state