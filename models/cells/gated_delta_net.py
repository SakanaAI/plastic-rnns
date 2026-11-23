import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.cells.base import RecurrentCell


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight).to(x.dtype)


class GatedDeltaNetCell(RecurrentCell):
    """Gated Delta Net, from "Gated Delta Networks: Improving Mamba2 with Delta Rule"."""

    def __init__(
        self,
        d_input: int,
        d_model: int,
        num_heads: int = 4,
        expand_k: float = 0.75,
        expand_v: float = 1.5,
        conv_size: int = 4,
        qk_norm: str = "l2",
        dropout: float = 0.0,
    ):
        super().__init__(d_input=d_input, d_model=d_model)
        assert qk_norm in ("l2", "softmax"), f"qk_norm must be 'l2' or 'softmax', got '{qk_norm}'"
        self.num_heads = num_heads
        self.key_dim = round(d_model * expand_k / num_heads) * num_heads
        self.value_dim = round(d_model * expand_v / num_heads) * num_heads
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.conv_size = conv_size
        self.qk_norm = qk_norm

        # Projections
        self.q_proj = nn.Linear(d_input, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_input, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_input, self.value_dim, bias=False)
        self.g_proj = nn.Linear(d_input, self.value_dim, bias=False)
        self.gk_proj = nn.Linear(d_input, num_heads, bias=False)
        self.b_proj = nn.Linear(d_input, num_heads, bias=True)
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)

        # Causal depthwise-convolution weights, one kernel per channel
        self.q_conv_weight = nn.Parameter(torch.empty(self.key_dim, conv_size))
        self.k_conv_weight = nn.Parameter(torch.empty(self.key_dim, conv_size))
        self.v_conv_weight = nn.Parameter(torch.empty(self.value_dim, conv_size))

        self.g_norm = RMSNorm(self.head_v_dim)

        # Mamba-style decay gating
        A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        # Initial states are always zero.
        self.register_buffer("S_0", torch.zeros(num_heads, self.head_qk_dim, self.head_v_dim))
        self.register_buffer("conv_q_0", torch.zeros(1, self.key_dim, conv_size))
        self.register_buffer("conv_k_0", torch.zeros(1, self.key_dim, conv_size))
        self.register_buffer("conv_v_0", torch.zeros(1, self.value_dim, conv_size))

        self.dropout_layer = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.g_proj, self.o_proj):
            nn.init.xavier_uniform_(proj.weight)
        nn.init.xavier_uniform_(self.gk_proj.weight)
        nn.init.xavier_uniform_(self.b_proj.weight)
        nn.init.zeros_(self.b_proj.bias)
        for weight in (self.q_conv_weight, self.k_conv_weight, self.v_conv_weight):
            nn.init.kaiming_normal_(weight.unsqueeze(1))

    def _conv_step(self, x: Tensor, conv_state: Tensor, conv_weight: Tensor) -> tuple[Tensor, Tensor]:
        """Apply one step of a causal depthwise convolution followed by SiLU."""
        new_state = torch.cat([conv_state[:, :, 1:], x.unsqueeze(-1)], dim=-1)
        out = (new_state * conv_weight).sum(-1)
        return F.silu(out), new_state

    @staticmethod
    def _gated_delta_update(S: Tensor, k: Tensor, v: Tensor, beta: Tensor, decay: Tensor) -> Tensor:
        """Apply Equation 10 of the Gated Delta Networks paper."""
        S_decayed = decay * S
        retrieved = torch.einsum("bhkv,bhk->bhv", S_decayed, k)
        delta = v - retrieved
        write = torch.einsum("bhk,bhv->bhkv", k, beta.unsqueeze(-1) * delta)
        return S_decayed + write

    def initialize_state(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.S_0.unsqueeze(0), self.conv_q_0, self.conv_k_0, self.conv_v_0

    def forward(
        self,
        input: Tensor,
        state: tuple[Tensor, Tensor, Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:  # noqa: A002
        assert input.ndim == 2, "Input should be of size B x D"
        state = state if state is not None else self.initialize_state()
        S, conv_q, conv_k, conv_v = self._expand_state(state, input.size(0))
        B = input.size(0)
        H, Dk, Dv = self.num_heads, self.head_qk_dim, self.head_v_dim

        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)

        q, conv_q = self._conv_step(q, conv_q, self.q_conv_weight)
        k, conv_k = self._conv_step(k, conv_k, self.k_conv_weight)
        v, conv_v = self._conv_step(v, conv_v, self.v_conv_weight)

        # The decay gate is represented in log space and is always negative.
        gk = self.gk_proj(input).float()
        gk = -self.A_log.float().exp() * F.softplus(gk + self.dt_bias)
        beta = self.b_proj(input).float().sigmoid()

        q = q.view(B, H, Dk)
        k = k.view(B, H, Dk)
        v = v.view(B, H, Dv)

        if self.qk_norm == "l2":
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        else:
            q, k = q.softmax(dim=-1), k.softmax(dim=-1)

        # Gated delta rule
        decay = gk.exp().unsqueeze(-1).unsqueeze(-1)
        S_new = self._gated_delta_update(S, k, v, beta, decay)

        # Read and apply the output gate.
        output = torch.einsum("bhkv,bhk->bhv", S_new, q) * (Dk ** -0.5)
        gate = self.g_proj(input).view(B, H, Dv)
        output = self.g_norm(output) * F.silu(gate)
        output = self.o_proj(output.reshape(B, self.value_dim))
        output = self.dropout_layer(output)

        return output, (S_new, conv_q, conv_k, conv_v)


class CrossAttentionGatedDeltaNetCell(GatedDeltaNetCell):
    """GDN state variant that exposes its previous output as an attention query."""

    def initialize_state(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        S, conv_q, conv_k, conv_v = super().initialize_state()
        query = torch.zeros(1, self.d_model, device=S.device, dtype=S.dtype)
        return S, conv_q, conv_k, conv_v, query

    def forward(
        self,
        input: Tensor,
        state: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:  # noqa: A002
        state = state if state is not None else self.initialize_state()
        S, conv_q, conv_k, conv_v, _ = state
        output, (S_new, conv_q, conv_k, conv_v) = super().forward(input, (S, conv_q, conv_k, conv_v))
        return output, (S_new, conv_q, conv_k, conv_v, output)
