import math

import torch
from torch import Tensor, nn

from models.cells.base import RecurrentCell


# Short-term plasticity neuron, from "Short-Term Plasticity Neurons Learning to Learn and Forget"
class STPNCell(RecurrentCell):
  def __init__(self, d_input: int, d_model: int, layer_norm: bool = True, dropout: float = 0.0):
    super().__init__(d_input=d_input, d_model=d_model)
    self.layer_norm = layer_norm
    self.h_0, self.F_0 = nn.Parameter(torch.zeros(d_model)), nn.Buffer(torch.zeros(d_input + d_model, d_model))  # Learnable initial hidden state
    self.weight = nn.Parameter(torch.empty(d_input + d_model, d_model))
    self.bias = nn.Parameter(torch.empty(d_model))
    self.Gamma = nn.Parameter(torch.empty(d_input + d_model, d_model))  # Hebbian learning rate
    self.Lambda = nn.Parameter(torch.empty(d_input + d_model, d_model))  # Hebbian forget rate
    self.dropout = nn.Dropout(p=dropout)
    if layer_norm:
      self.ln = nn.LayerNorm(d_model)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.orthogonal_(self.weight)
    nn.init.zeros_(self.bias)
    nn.init.uniform_(self.Gamma, a=-0.001 / math.sqrt(self.d_model), b=0.001 / math.sqrt(self.d_model))
    nn.init.uniform_(self.Lambda, a=0, b=1)

  def initialize_state(self) -> tuple[Tensor, Tensor]:
    return self.h_0.unsqueeze(0), self.F_0.unsqueeze(0)

  def forward(self, input: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]:  # noqa: A002
    assert input.ndim == 2, "Input should be of size B x D"
    h_0, F_0 = self.dropout(state[0].expand(input.shape[0], -1)), state[1].expand(input.shape[0], -1, -1)  # Expand initial state to batch size; no-op otherwise

    G_1 = self.weight.unsqueeze(dim=0) + F_0  # Total efficacy G_t = W + F_t-1
    G_1_norm = torch.linalg.norm(G_1, ord=2, dim=1, keepdim=True)  # Calculate normalisation per neuron
    G_1, F_0 = G_1 / G_1_norm, F_0 / G_1_norm  # Normalise total efficacy and short-term component
    x_1_h_0 = torch.cat([input, h_0], dim=1)
    if self.layer_norm:  # noqa: SIM108
      h_1 = torch.tanh(self.ln(torch.einsum("bxh,bx->bh", G_1, x_1_h_0) + self.bias.unsqueeze(0)))  # Forward pass h_t = σ(LN(G_t⋅[x_t; d(h_t−1)] + b))
    else:
      h_1 = torch.tanh(torch.einsum("bxh,bx->bh", G_1, x_1_h_0) + self.bias.unsqueeze(0))  # Forward pass h_t = σ(G_t⋅[x_t; d(h_t−1)] + b)
    F_1 = self.Gamma * torch.einsum("bx,bh->bxh", x_1_h_0, h_1) + self.Lambda * F_0  # Hebbian STP update F_t = Γ ⊙ ([x_t; d(h_t−1)] ⊗ h_t) + Λ ⊙ F_t-1
    return h_1, (h_1, F_1)