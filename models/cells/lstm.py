import torch
from torch import Tensor, nn

from models.cells.base import RecurrentCell

class LSTMCell(RecurrentCell):
  def __init__(self, d_input: int, d_model: int, dropout: float = 0, layer_norm: bool = True, T_max: int | None = None):
    super().__init__(d_input=d_input, d_model=d_model)
    assert 0 <= dropout < 1, "Dropout should be >= 0 and < 1"
    assert T_max is None or T_max >= 2, "T_max should be >= 2 if set"
    self.layer_norm, self.T_max = layer_norm, T_max
    self.h_0, self.c_0 = nn.Parameter(torch.zeros(d_model)), nn.Parameter(torch.zeros(d_model))  # Learnable initial hidden states
    self.weight_h = nn.Parameter(torch.empty(d_model, 4 * d_model))
    self.weight_x = nn.Parameter(torch.empty(d_input, 4 * d_model))
    self.bias = nn.Parameter(torch.empty(4 * d_model))
    self.dropout = nn.Dropout(p=dropout)  # Recurrent dropout for hidden state updates, from "Recurrent Dropout without Memory Loss"
    if layer_norm:
      self.ln_1 = nn.LayerNorm(4 * d_model)  # First LayerNorm configuration from "Layer Normalization"
      self.ln_2 = nn.LayerNorm(4 * d_model)
      self.ln_3 = nn.LayerNorm(d_model)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.orthogonal_(self.weight_x)
    nn.init.orthogonal_(self.weight_h)
    nn.init.zeros_(self.bias)
    if self.T_max:
      bias_f = torch.log(torch.zeros(self.d_model).uniform_(1, self.T_max - 1))  # If T_max is given, use chrono initialisation from "Can recurrent neural networks warp time?"
      self.bias.data[: self.d_model], self.bias.data[self.d_model : 2 * self.d_model] = bias_f, -bias_f
    else:
      nn.init.ones_(self.bias[: self.d_model])  # Initialise high forget gate bias, from "An Empirical Exploration of Recurrent Network Architectures"

  def _calculate_gates(self, input: Tensor, h_0: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # noqa: A002
    if self.layer_norm:  # noqa: SIM108
      gates = self.ln_1(h_0 @ self.weight_h) + self.ln_2(input @ self.weight_x) + self.bias.unsqueeze(0)  # f_t, i_t, o_t, g_t = LN(W_h⋅h_t−1; α_1, β_1) + LN(W_x⋅x_t; α_2, β_2) + b
    else:
      gates = h_0 @ self.weight_h + input @ self.weight_x + self.bias.unsqueeze(0)  # f_t, i_t, o_t, g_t = W_h⋅h_t−1 + W_x⋅x_t + b
    return gates.chunk(4, dim=1)

  def _calculate_cell_hidden_updates(self, c_0: Tensor, forget_gate: Tensor, in_gate: Tensor, out_gate: Tensor, cell_gate: Tensor) -> tuple[Tensor, Tensor]:
    c_1 = torch.sigmoid(forget_gate) * c_0 + torch.sigmoid(in_gate) * self.dropout(torch.tanh(cell_gate))  # c_t = σ(f_t) ⊙ c_t−1 + σ(i_t) ⊙ d(tanh(g_t))
    if self.layer_norm:  # noqa: SIM108
      h_1 = torch.sigmoid(out_gate) * torch.tanh(self.ln_3(c_1))  # h_t = σ(o_t) ⊙ tanh(LN(c_t; α_3, β_3))
    else:
      h_1 = torch.sigmoid(out_gate) * torch.tanh(c_1)  # h_t = σ(o_t) ⊙ tanh(c_t)
    return h_1, c_1

  def initialize_state(self):
    h_0 = self.h_0.unsqueeze(0)
    c_0 = self.c_0.unsqueeze(0)
    return (h_0, c_0)
  
  def forward(self, input: Tensor, state: tuple[Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor]]:  # noqa: A002
    assert input.ndim == 2, "Input should be of size B x D"
    h_0, c_0 = state
    forget_gate, in_gate, out_gate, cell_gate = self._calculate_gates(input, h_0)
    h_1, c_1 = self._calculate_cell_hidden_updates(c_0, forget_gate, in_gate, out_gate, cell_gate)
    return h_1, (h_1, c_1)

# Plastic LSTM, from "Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity"
class PlasticLSTMCell(LSTMCell):  # Change from RecurrentCell to LSTMCell
  def __init__(self, d_input: int, d_model: int, dropout: float = 0, layer_norm: bool = True, T_max: int | None = None, neuromodulation: str | None = None):
    super().__init__(d_input, d_model, dropout=dropout, layer_norm=layer_norm, T_max=T_max)
    assert neuromodulation is None or neuromodulation in ["simple", "retroactive"], "Neuromodulation should either be none or 'simple' or 'retroactive'"
    self.neuromodulation = neuromodulation
    self.Hebb_0 = nn.Buffer(torch.zeros(d_model, d_model))  # Add buffer
    if neuromodulation == "retroactive":
        self.E_0 = nn.Buffer(torch.zeros(d_model, d_model))  # Add buffer
    self.alpha = nn.Parameter(torch.empty(d_model, d_model))
    if neuromodulation:
      self.M = nn.Sequential(nn.Linear(d_model, 1), nn.Tanh(), nn.Linear(1, d_model))  # Neuromodulation module (bottleneck)
    if neuromodulation is None or neuromodulation == "retroactive":
      self.eta = nn.Parameter(torch.empty((1, )))  # Plasticity/eligibility trace learning rate
    self.reset_parameters()

  def reset_parameters(self):
    super().reset_parameters()
    if hasattr(self, "neuromodulation"):
      nn.init.uniform_(self.alpha, a=0, b=0.01)
      if self.neuromodulation:
        nn.init.orthogonal_(self.M[0].weight)
        nn.init.zeros_(self.M[0].bias)
        nn.init.orthogonal_(self.M[2].weight)
        nn.init.zeros_(self.M[2].bias)
      if self.neuromodulation is None or self.neuromodulation == "retroactive":
        nn.init.constant_(self.eta, 0.01)

  def initialize_state(self):
      h_0 = self.h_0.unsqueeze(0)
      c_0 = self.c_0.unsqueeze(0)
      Hebb_0 = self.Hebb_0.unsqueeze(0)
      if self.neuromodulation == "retroactive":
          E_0 = self.E_0.unsqueeze(0)
          return (h_0, c_0, Hebb_0, E_0)
      else:
          return (h_0, c_0, Hebb_0)

  def _calculate_gates(self, input: Tensor, h_0: Tensor, Hebb_0: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # noqa: A002
    weight_h = self.weight_h.unsqueeze(dim=0) + torch.cat([torch.zeros(input.shape[0], self.d_model, 3 * self.d_model, device=input.device), self.alpha.unsqueeze(dim=0) * Hebb_0], dim=2)  # Add plastic weights for the cell gate: W_h + α⋅Hebb_t-1
    if self.layer_norm:  # noqa: SIM108
      gates = self.ln_1(torch.einsum("bi,bih->bh", h_0, weight_h)) + self.ln_2(input @ self.weight_x) + self.bias.unsqueeze(0)  # f_t, i_t, o_t, g_t = LN((W_h + α⋅Hebb_t-1)⋅h_t−1; α_1, β_1) + LN(W_x⋅x_t; α_2, β_2) + b
    else:
      gates = torch.einsum("bi,bih->bh", h_0, weight_h) + input @ self.weight_x + self.bias.unsqueeze(0)  # f_t, i_t, o_t, g_t = (W_h + α⋅Hebb_t-1)⋅h_t−1 + W_x⋅x_t + b
    return gates.chunk(4, dim=1)

  def forward(self, input: Tensor, state: tuple[Tensor, Tensor, ...] | None) -> tuple[Tensor, tuple[Tensor, Tensor, ...]]:
      assert input.ndim == 2, "Input should be of size B x D"
      B = input.shape[0]
      if self.neuromodulation == "retroactive":
        h_0, c_0, Hebb_0, E_0 = state[0], state[1], state[2].expand(B, -1, -1), state[3].expand(B, -1, -1)
      else:
        h_0, c_0, Hebb_0 = state[0], state[1], state[2].expand(B, -1, -1)

      forget_gate, in_gate, out_gate, cell_gate = self._calculate_gates(input, h_0, Hebb_0)
      h_1, c_1 = self._calculate_cell_hidden_updates(c_0, forget_gate, in_gate, out_gate, cell_gate)

      if self.neuromodulation == "simple":
        Hebb_1 = torch.clip(Hebb_0 + self.M(h_0).unsqueeze(1) * torch.einsum("bh,bi->bhi", h_0, torch.tanh(cell_gate)), min=-1, max=1)
        return h_1, (h_1, c_1, Hebb_1)
      elif self.neuromodulation == "retroactive":
        Hebb_1 = torch.clip(Hebb_0 + self.M(h_0).unsqueeze(1) * E_0, min=-1, max=1)
        E_1 = (1 - self.eta) * E_0 + self.eta * torch.einsum("bh,bi->bhi", h_0, torch.tanh(cell_gate))
        return h_1, (h_1, c_1, Hebb_1, E_1)
      else:
        Hebb_1 = torch.clip(Hebb_0 + self.eta * torch.einsum("bh,bi->bhi", h_0, torch.tanh(cell_gate)), min=-1, max=1)
        return h_1, (h_1, c_1, Hebb_1)


# HyperLSTM, from "HyperNetworks"
class HyperLSTMCell(RecurrentCell):
  def __init__(self, d_input: int, d_model: int, hyper_size: int, embedding_size: int, dropout: float = 0, layer_norm: bool = True):
    super().__init__(d_input=d_input, d_model=d_model)
    assert 0 <= dropout < 1, "Dropout should be >= 0 and < 1"
    self.embedding_size, self.layer_norm = embedding_size, layer_norm
    self.h_0, self.c_0, self.h_hat_0, self.c_hat_0 = nn.Parameter(torch.zeros(d_model)), nn.Parameter(torch.zeros(d_model)), nn.Parameter(torch.zeros(hyper_size)), nn.Parameter(torch.zeros(hyper_size))  # Learnable initial hidden states
    self.hweight = nn.Parameter(torch.empty(hyper_size + d_model + d_input, 4 * hyper_size))
    self.hbias = nn.Parameter(torch.empty(4 * hyper_size))
    if self.layer_norm:
      self.hln_i, self.hln_g, self.hln_f, self.hln_o, self.hln_c = [nn.LayerNorm(hyper_size) for _ in range(5)]
    self.zweight_h, self.zweight_x, self.zweight_b = [nn.Parameter(torch.empty(hyper_size, 4 * embedding_size)) for _ in range(3)]
    self.zbias_h, self.zbias_x = [nn.Parameter(torch.empty(4 * embedding_size)) for _ in range(2)]
    self.dweight_ih, self.dweight_gh, self.dweight_fh, self.dweight_oh = [nn.Parameter(torch.empty(embedding_size, d_model)) for _ in range(4)]
    self.dweight_ix, self.dweight_gx, self.dweight_fx, self.dweight_ox = [nn.Parameter(torch.empty(embedding_size, d_model)) for _ in range(4)]
    self.dweight_ib, self.dweight_gb, self.dweight_fb, self.dweight_ob = [nn.Parameter(torch.empty(embedding_size, d_model)) for _ in range(4)]
    self.weight = nn.Parameter(torch.empty(d_model + d_input, 4 * d_model))
    self.bias = nn.Parameter(torch.empty(4 * d_model))
    if self.layer_norm:
      self.ln_i, self.ln_g, self.ln_f, self.ln_o, self.ln_c = [nn.LayerNorm(d_model) for _ in range(5)]
    self.dropout = nn.Dropout(p=dropout)  # Recurrent dropout for hidden state updates, from "Recurrent Dropout without Memory Loss"
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.orthogonal_(self.hweight)
    nn.init.zeros_(self.hbias)
    nn.init.zeros_(self.zweight_h)
    nn.init.zeros_(self.zweight_x)
    nn.init.normal_(self.zweight_b, std=0.01)
    nn.init.ones_(self.zbias_h)
    nn.init.ones_(self.zbias_x)
    nn.init.constant_(self.dweight_ih, 0.1 / self.embedding_size)
    nn.init.constant_(self.dweight_gh, 0.1 / self.embedding_size)
    nn.init.constant_(self.dweight_fh, 0.1 / self.embedding_size)
    nn.init.constant_(self.dweight_oh, 0.1 / self.embedding_size)
    nn.init.constant_(self.dweight_ix, 0.1 / self.embedding_size)
    nn.init.constant_(self.dweight_gx, 0.1 / self.embedding_size)
    nn.init.constant_(self.dweight_fx, 0.1 / self.embedding_size)
    nn.init.constant_(self.dweight_ox, 0.1 / self.embedding_size)
    nn.init.zeros_(self.dweight_ib)
    nn.init.zeros_(self.dweight_gb)
    nn.init.zeros_(self.dweight_fb)
    nn.init.zeros_(self.dweight_ob)  
    nn.init.orthogonal_(self.weight)
    nn.init.zeros_(self.bias)

  def initialize_state(self):
      h_0 = self.h_0.unsqueeze(0)
      c_0 = self.c_0.unsqueeze(0)
      h_hat_0 = self.h_hat_0.unsqueeze(0)
      c_hat_0 = self.c_hat_0.unsqueeze(0)
      return (h_0, c_0, h_hat_0, c_hat_0)

  def forward(self, input: Tensor, state: tuple[Tensor, Tensor, Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:  # noqa: A002
    assert input.ndim == 2, "Input should be of size B x D"
    B = input.shape[0]
    h_0, c_0, h_hat_0, c_hat_0 = state[0].expand(B, -1), state[1], state[2].expand(B, -1), state[3]

    h_hat_1_x_hat_1 = torch.cat([h_hat_0, h_0, input], dim=1)  # x^_t = [h_t-1; x_t]
    input_gate_hat, cell_gate_hat, forget_gate_hat, output_gate_hat = (h_hat_1_x_hat_1 @ self.hweight + self.hbias.unsqueeze(0)).chunk(4, dim=1)  # i^_t, g^_t, f^_t, o^_t = W_y^h^⋅h^_t−1 + W_y^x^⋅x^_t + b_y^
    if self.layer_norm:
      input_gate_hat, cell_gate_hat, forget_gate_hat, output_gate_hat = self.hln_i(input_gate_hat), self.hln_g(cell_gate_hat), self.hln_f(forget_gate_hat), self.hln_o(output_gate_hat)  # i^_t, g^_t, f^_t, o^_t = LN(y^_t)
    c_hat_1 = torch.sigmoid(forget_gate_hat) * c_hat_0 + torch.sigmoid(input_gate_hat) * torch.tanh(cell_gate_hat)  # c^_t = σ(f^_t) ⊙ c^_t−1 + σ(i^_t) ⊙ tanh(g^_t)
    if self.layer_norm:
      h_hat_1 = torch.sigmoid(output_gate_hat) * torch.tanh(self.hln_c(c_hat_1))  # h^_t = σ(o^_t) ⊙ tanh(LN(c^_t))
    else:
      h_hat_1 = torch.sigmoid(output_gate_hat) * torch.tanh(c_hat_1)  # h^_t = σ(o^_t) ⊙ tanh(c^_t)

    z_ih, z_gh, z_fh, z_oh = (h_hat_1 @ self.zweight_h + self.zbias_h.unsqueeze(0)).chunk(4, dim=1)  # z_ih, z_gh, z_fh, z_oh = W_yh^h⋅h^_t−1 + b_yh^h
    z_ix, z_gx, z_fx, z_ox = (h_hat_1 @ self.zweight_x + self.zbias_x.unsqueeze(0)).chunk(4, dim=1)  # z_ix, z_gx, z_fx, z_ox = W_yh^x⋅h^_t−1 + b_yh^x
    z_ib, z_gb, z_fb, z_ob = (h_hat_1 @ self.zweight_b).chunk(4, dim=1)  # z_ib, z_gb, z_fb, z_ob = W_yh^b⋅h^_t−1

    d_ih, d_gh, d_fh, d_oh = z_ih @ self.dweight_ih, z_gh @ self.dweight_gh, z_fh @ self.dweight_fh, z_oh @ self.dweight_oh  # d_ih, d_gh, d_fh, d_oh = W_yhz⋅z_yh
    d_ix, d_gx, d_fx, d_ox = z_ix @ self.dweight_ix, z_gx @ self.dweight_gx, z_fx @ self.dweight_fx, z_ox @ self.dweight_ox  # d_ix, d_gx, d_fx, d_ox = W_yxz⋅z_yx
    d_ib, d_gb, d_fb, d_ob = z_ib @ self.dweight_ib, z_gb @ self.dweight_gb, z_fb @ self.dweight_fb, z_ob @ self.dweight_ob  # d_ib, d_gb, d_fb, d_ob = W_ybz⋅z_yb

    d_yh, d_yx, d_yb = torch.cat([d_ih, d_gh, d_fh, d_oh], dim=1), torch.cat([d_ix, d_gx, d_fx, d_ox], dim=1), torch.cat([d_ib, d_gb, d_fb, d_ob], dim=1)
    input_gate, cell_gate, forget_gate, output_gate= ((torch.cat([h_0, input], dim=1) @ self.weight) * d_yh * d_yx + d_yb + self.bias.unsqueeze(0)).chunk(4, dim=1)  # i_t, g_t, f_t, o_t = d_yh ⊙ W_yh⋅h_t−1 + d_yx ⊙ W_yx⋅x_t + d_yb + b_y
    if self.layer_norm:
      input_gate, cell_gate, forget_gate, output_gate= self.ln_i(input_gate), self.ln_g(cell_gate), self.ln_f(forget_gate), self.ln_o(output_gate)  # i_t, g_t, f_t, o_t = LN(y_t)
    c_1 = torch.sigmoid(forget_gate) * c_0 + torch.sigmoid(input_gate) * self.dropout(torch.tanh(cell_gate))  # c_t = σ(f_t) ⊙ c_t−1 + σ(i_t) ⊙ d(tanh(g_t))
    if self.layer_norm:
      h_1 = torch.sigmoid(output_gate) * torch.tanh(self.ln_c(c_1))  # h_t = σ(o_t) ⊙ tanh(LN(c_t))
    else:
      h_1 = torch.sigmoid(output_gate) * torch.tanh(c_1)  # h_t = σ(o_t) ⊙ tanh(c_t)
    return h_1, (h_1, c_1, h_hat_1, c_hat_1)
