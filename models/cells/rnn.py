import torch
from torch import Tensor, nn

from models.cells.base import RecurrentCell

class RNNCell(RecurrentCell):
  def __init__(self, d_input: int, d_model: int, dropout: float = 0.0, layer_norm: bool = True):
    super().__init__(d_input=d_input, d_model=d_model)
    self.h_0 = nn.Parameter(torch.zeros(d_model))  # Learnable initial hidden state
    self.weight_x = nn.Parameter(torch.empty(d_input, d_model))
    self.weight_h = nn.Parameter(torch.empty(d_model, d_model))
    self.bias = nn.Parameter(torch.empty(d_model))
    self.layer_norm = layer_norm
    self.dropout = nn.Dropout(p=dropout)
    if layer_norm:
      self.ln = nn.LayerNorm(d_model)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.orthogonal_(self.weight_x)
    nn.init.orthogonal_(self.weight_h)
    nn.init.zeros_(self.bias)

  def initialize_state(self) -> tuple[Tensor]:
    return (self.h_0.unsqueeze(0),)

  def forward(self, input: Tensor, state: tuple[Tensor] | None) -> tuple[Tensor, tuple[Tensor]]:  # noqa: A002
    assert input.ndim == 2, "Input should be of size B x D"
    h_0 = self.dropout(state[0]) if state is not None else self.h_0.unsqueeze(0)

    if self.layer_norm:  # noqa: SIM108
      h_1 = torch.tanh(self.ln(input @ self.weight_x + h_0 @ self.weight_h + self.bias.unsqueeze(0)))  # h_t = tanh(LN(W_x⋅x_t + W_h⋅d(h_t−1) + b))
    else:
      h_1 = torch.tanh(input @ self.weight_x + h_0 @ self.weight_h + self.bias.unsqueeze(0))  # h_t = tanh(W_x⋅x_t + W_h⋅d(h_t−1) + b)
    return h_1, (h_1,)

# Plastic RNN, from "Differentiable plasticity: training plastic neural networks with backpropagation"
class PlasticRNNCell(RNNCell):
  def __init__(self, d_input: int, d_model: int, dropout: float = 0.0, layer_norm: bool = True, rule: str = "hebb"):
    super().__init__(d_input=d_input, d_model=d_model, dropout=dropout, layer_norm=layer_norm)
    assert rule in ["hebb", "oja"], "Plasticity rule must be either 'hebb' or 'oja'"
    self.hidden_size, self.rule = d_model, rule
    self.Hebb_0 = nn.Buffer(torch.zeros(d_model, d_model))
    self.alpha = nn.Parameter(torch.empty(d_model, d_model))  # Plasticity coefficient
    self.eta = nn.Parameter(torch.empty((1,)))  # Plasticity learning rate
    self.reset_parameters()

  def reset_parameters(self):
    super().reset_parameters()
    if hasattr(self, "alpha"):
      nn.init.uniform_(self.alpha, a=0, b=0.01)
      nn.init.constant_(self.eta, 0.01)

  def initialize_state(self) -> tuple[Tensor, Tensor]:
    return self.h_0.unsqueeze(0), self.Hebb_0.unsqueeze(0)

  def forward(self, input: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]:  # noqa: A002
    assert input.ndim == 2, "Input should be of size B x D"
    h_0, Hebb_0 = self.dropout(state[0]), state[1]

    if self.layer_norm:  # noqa: SIM108
      h_1 = torch.tanh(self.ln(input @ self.weight_x + torch.einsum("bh,bih->bh", h_0, self.weight_h.unsqueeze(0) + self.alpha * Hebb_0) + self.bias.unsqueeze(0)))  # h_t = tanh(LN(W_x⋅x_t + (W_h + α⋅Hebb_t-1)⋅d(h_t−1) + b))
    else:
      h_1 = torch.tanh(input @ self.weight_x + torch.einsum("bh,bih->bh", h_0, self.weight_h.unsqueeze(0) + self.alpha * Hebb_0) + self.bias.unsqueeze(0))  # h_t = tanh(W_x⋅x_t + (W_h + α⋅Hebb_t-1)⋅d(h_t−1) + b)

    if self.rule == "hebb":
      Hebb_1 = self.eta * torch.einsum("bi,bj->bij", h_0, h_1) + (1 - self.eta) * Hebb_0  # Hebb_i,j_t = η⋅x_i_t-1 ⊗ x_j_t + (1 - η)⋅Hebb_i,j_t-1
    else:
      Hebb_1 = Hebb_0 + self.eta * (torch.einsum("bi,bj->bij", h_0, h_1) - (h_1 ** 2).unsqueeze(1) * Hebb_0)  # Hebb_i,j_t = Hebb_i,j_t-1 + η⋅(x_i_t-1⋅x_j_t - (x_j_t)²⋅Hebb_i,j_t-1)
    return h_1, (h_1, Hebb_1)
