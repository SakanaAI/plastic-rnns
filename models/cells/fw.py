import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.cells.rnn import RNNCell


# Fast weights, from "Using Fast Weights to Attend to the Recent Past"
class FW(RNNCell):
  def __init__(self, d_input: int, d_model: int, lmbda: float = 0.9, eta: float = 0.5, steps: int = 1, layer_norm: bool = True, dropout: float = 0.0):
    super().__init__(d_input=d_input, d_model=d_model, dropout=dropout, layer_norm=layer_norm)
    self.steps = steps
    self.A_0 = nn.Buffer(torch.zeros(d_model, d_model))
    self.lmbda = nn.Buffer(torch.tensor(lmbda))  # Fast weights decay rate
    self.eta = nn.Buffer(torch.tensor(eta))  # Fast weights learning rate
    self.reset_parameters()

  def reset_parameters(self):
    if hasattr(self, "lmbda"):
      nn.init.uniform_(self.weight_x, -1 / math.sqrt(self.d_model), 1 / math.sqrt(self.d_model))
      nn.init.eye_(self.weight_h)
      self.weight_h.data.mul_(0.05)
      nn.init.zeros_(self.bias)

  def initialize_state(self) -> tuple[Tensor, Tensor]:
    return self.h_0.unsqueeze(0), self.A_0.unsqueeze(0)

  def forward(self, input: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]:  # noqa: A002
    assert input.ndim == 2, "Input should be of size B x D"
    h_0, A_0 = self.dropout(state[0]), state[1]

    slow_pre_hidden = input @ self.weight_x + h_0 @ self.weight_h + self.bias.unsqueeze(0)  # Calculate slow weight components (sustained boundary condition) W_x⋅x_t + W_h⋅d(h_t−1) + b
    if self.layer_norm:  # noqa: SIM108
      h_1 = F.relu(self.ln(slow_pre_hidden))  # Calculate preliminary internal hidden state h_s = ReLU(LN(W_x⋅x_t + W_h⋅d(h_t−1) + b))
    else:
      h_1 = F.relu(slow_pre_hidden)  # Calculate preliminary internal hidden state h_s = ReLU(W_x⋅x_t + W_h⋅d(h_t−1) + b)
    for _ in range(self.steps):
      if self.layer_norm:  # noqa: SIM108
        h_1 = F.relu(self.ln(slow_pre_hidden + torch.einsum("bij,bj->bi", A_0, h_1)))  # Refine internal hidden state h_s = ReLU(LN(W_x⋅x_t + W_h⋅d(h_t−1) + A_t-1⋅h_s + b))
      else:
        h_1 = F.relu(slow_pre_hidden + torch.einsum("bij,bj->bi", A_0, h_1))  # Refine internal hidden state h_s = ReLU(W_x⋅x_t + W_h⋅d(h_t−1) + A_t-1⋅h_s + b)

    A_1 = self.lmbda * A_0 + self.eta * torch.einsum("bi,bj->bij", h_1, h_1)  # A_t = λ⋅A_t−1 + η⋅h_t ⊗ h_t
    return h_1, (h_1, A_1)