import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parametrize

from models.cells.base import RecurrentCell

class Bounded(nn.Module):
  def __init__(self, low: float, high: float):
    super().__init__()
    self.low, self.high = low, high

  def forward(self, x: Tensor) -> Tensor:
    return (self.high - self.low) * torch.sigmoid(x) + self.low

  def right_inverse(self, y: Tensor) -> Tensor:
    y_normalised = torch.clamp((y - self.low) / (self.high - self.low), 1e-6, 1 - 1e-6)
    return torch.logit(y_normalised)

class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

class SuperLinear(nn.Module):
    def __init__(self, in_dims, out_dims, N, dropout=0, init_identity=False):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.in_dims = in_dims
        self.init_identity = init_identity
        self.register_parameter('w1', nn.Parameter(torch.empty((in_dims, out_dims, N))))
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_identity:
            nn.init.zeros_(self.w1)
            with torch.no_grad():
                self.w1.data[-1, 0, :] = 2.0
        else:
            bound = 1 / math.sqrt(self.in_dims + self.w1.shape[1])
            nn.init.uniform_(self.w1, -bound, bound)

    def forward(self, x):
        out = self.dropout(x)
        out = torch.einsum('BDM,MHD->BDH', out, self.w1)
        out = out.squeeze(-1)
        return out

class PlasticLinear(nn.Module):
    def __init__(self, hidden_features, init_beta=0.3, normalize_weights=True, adaptive_scaling=True, fixed_scale_factor=0.3):
        super().__init__()
        self.init_beta = init_beta
        self.normalize_weights = normalize_weights
        self.adaptive_scaling = adaptive_scaling
        self.fixed_scale_factor = fixed_scale_factor
        self.weights_fixed = nn.Parameter(torch.empty(hidden_features, hidden_features))
        self.proj_to_d_model = nn.LazyLinear(hidden_features)
        if adaptive_scaling:
            self.beta = nn.Parameter(torch.tensor(init_beta))
            parametrize.register_parametrization(self, "beta", Bounded(0.0, 2.0))
        self.layernorm = nn.LayerNorm(hidden_features)
        nn.init.orthogonal_(self.weights_fixed)

    def reset_parameters(self):
        nn.init.orthogonal_(self.weights_fixed)
        self.proj_to_d_model.reset_parameters()
        with torch.no_grad():
            if self.adaptive_scaling:
                self.beta = torch.full_like(self.beta, self.init_beta)

    def forward(self, x, weights_plastic=None):
        B = x.shape[0]

        weights_fixed = self.weights_fixed.unsqueeze(0).expand(B, -1, -1)

        if weights_plastic is not None:
            if self.adaptive_scaling:
                weights_plastic_norm = torch.norm(weights_plastic, p='fro', dim=(-2, -1), keepdim=True)
                weights_fixed_norm = torch.norm(weights_fixed, p='fro', dim=(-2, -1), keepdim=True)
                alpha = weights_fixed_norm * self.beta / (weights_plastic_norm + 1e-6) # Adaptive scaling
                weights = weights_fixed + alpha * weights_plastic
            else:
                weights = weights_fixed + self.fixed_scale_factor * weights_plastic
        else:
            weights = weights_fixed

        if self.normalize_weights:
            weights = weights / (torch.norm(weights, p='fro', dim=(-2, -1), keepdim=True) + 1e-6)

        output = torch.bmm(self.proj_to_d_model(x).unsqueeze(1), weights).squeeze(1)
        output = self.layernorm(output)

        return output

class HebbianPlasticLinear(nn.Module):
    def __init__(self, hidden_features):
        super().__init__()
        self.weights_fixed = nn.Parameter(torch.empty(hidden_features, hidden_features))
        self.proj_to_d_model = nn.LazyLinear(hidden_features)
        self.alpha = nn.Parameter(torch.empty(hidden_features, hidden_features))
        self.layernorm = nn.LayerNorm(hidden_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weights_fixed)
        nn.init.uniform_(self.alpha, a=0, b=0.01)

    def forward(self, x, weights_plastic=None):
        B = x.shape[0]

        weights_fixed = self.weights_fixed.unsqueeze(0).expand(B, -1, -1)

        if weights_plastic is not None:
            weights = weights_fixed + self.alpha * weights_plastic
        else:
            weights = weights_fixed

        weights_normalized = weights / (torch.norm(weights, p='fro', dim=(-2, -1), keepdim=True) + 1e-6)

        output = torch.bmm(self.proj_to_d_model(x).unsqueeze(1), weights_normalized).squeeze(1)
        output = self.layernorm(output)

        return output

class Synchronizer(nn.Module):
    def __init__(self, d_model: int, sync_repr_size: int):
        super().__init__()
        self._set_left_right_neuron_idxs(d_model, sync_repr_size)
        self.r = nn.Parameter(torch.empty(sync_repr_size))
        self.rmsn = nn.RMSNorm(sync_repr_size, elementwise_affine=True)
        nn.init.uniform_(self.r, 0, 1)
        parametrize.register_parametrization(self, "r", Bounded(0, 15))

    def reset_parameters(self):
        with torch.no_grad():
            self.r = torch.empty_like(self.r).uniform_(0, 1)

    def _set_left_right_neuron_idxs(self, d_model: int, sync_repr_size: int):
        total_pairs = d_model ** 2
        assert sync_repr_size <= total_pairs, f"Synchronisation representation size ({sync_repr_size}) cannot be larger than the total number of possible pairs ({total_pairs})"
        pair_idxs = torch.randperm(total_pairs)[:sync_repr_size].sort()[0]
        self.neuron_idxs_left, self.neuron_idxs_right = nn.Buffer(pair_idxs // d_model), nn.Buffer(pair_idxs % d_model)

    def forward(self, z: Tensor, alpha_0: Tensor, beta_0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z_z = z[:, self.neuron_idxs_left] * z[:, self.neuron_idxs_right]

        exp_minus_r = torch.exp(-self.r).unsqueeze(0).repeat(z.shape[0], 1)
        alpha_1, beta_1 = exp_minus_r * alpha_0 + z_z, exp_minus_r * beta_0 + 1  # α_t = e^-r⋅α_t-1 + z_i_t ⊗ z_j_t; β_t = e^-r⋅β_t-1 + 1

        S = alpha_1 / (torch.sqrt(beta_1) + 1e-8)  # S_t = α_t/√︃(β_t)
        S = self.rmsn(S)
        return S, alpha_1, beta_1


class _CTMCellBase(RecurrentCell):
    """Common infrastructure shared by all CTM cell variants."""

    def __init__(self, d_input, d_model, memory_length, dropout, plastic=False, plastic_normalize_weights=True, plastic_adaptive_scaling=True, plastic_fixed_scale_factor=0.3):
        super().__init__(d_input=d_input, d_model=d_model)
        self.memory_length = memory_length
        self.plastic = plastic
        self.sync_repr_size = d_model ** 2 if plastic else d_model
        self.synapses = PlasticLinear(hidden_features=d_model, normalize_weights=plastic_normalize_weights, adaptive_scaling=plastic_adaptive_scaling, fixed_scale_factor=plastic_fixed_scale_factor)
        self.synchronizer = Synchronizer(d_model=d_model, sync_repr_size=self.sync_repr_size)
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        self.synapses.reset_parameters()
        self.synchronizer.reset_parameters()

    def _init_sync_state(self, activated_state):
        """Create initial synchronization state tensors."""
        decay_alpha = torch.zeros(1, self.sync_repr_size, device=activated_state.device)
        decay_beta = torch.zeros(1, self.sync_repr_size, device=activated_state.device)
        initial_synch = activated_state[:, self.synchronizer.neuron_idxs_left] * activated_state[:, self.synchronizer.neuron_idxs_right]
        return decay_alpha, decay_beta, initial_synch
    
class ContinuousThoughtMachineCell(_CTMCellBase):
    def __init__(self, d_input, d_model, memory_length, dropout, plastic=False, nlm_init_identity=False, plastic_normalize_weights=True, plastic_adaptive_scaling=True, plastic_fixed_scale_factor=0.3):
        super().__init__(d_input, d_model, memory_length, dropout, plastic, plastic_normalize_weights=plastic_normalize_weights, plastic_adaptive_scaling=plastic_adaptive_scaling, plastic_fixed_scale_factor=plastic_fixed_scale_factor)
        self.nlms = nn.Sequential(SuperLinear(in_dims=memory_length, out_dims=2, N=d_model, dropout=dropout, init_identity=nlm_init_identity), nn.GLU(), Squeeze(-1))
        self.start_trace = nn.Parameter(torch.zeros((d_model, memory_length)))
        self.start_activated_state = nn.Parameter(torch.zeros((d_model)))

    def reset_parameters(self):
        super().reset_parameters()
        self.nlms[0].reset_parameters()
        nn.init.zeros_(self.start_trace)
        nn.init.zeros_(self.start_activated_state)

    def initialize_state(self):
        state_trace = self.start_trace.unsqueeze(0) # Shape: (1, H, T)
        activated_state = self.start_activated_state.unsqueeze(0) # Shape: (1, H)
        decay_alpha, decay_beta, initial_synch = self._init_sync_state(activated_state)
        return (activated_state, state_trace, decay_alpha, decay_beta, initial_synch)

    def forward(self, input, state):
        assert input.ndim == 2, "Input should be of size B x D"
        B = input.shape[0]
        activated_state, state_trace, decay_alpha, decay_beta, synchronisation = self.dropout(state[0]), state[1], state[2], state[3], state[4]
        activated_state, state_trace, decay_alpha, decay_beta, synchronisation = self._expand_state((activated_state, state_trace, decay_alpha, decay_beta, synchronisation), B)

        state = self.synapses(torch.concat([input, activated_state], dim=-1), weights_plastic=synchronisation.view(B, self.d_model, self.d_model) if self.plastic else None)
        state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

        activated_state = self.nlms(state_trace)

        synchronisation, decay_alpha, decay_beta = self.synchronizer(activated_state, decay_alpha, decay_beta)

        return synchronisation[:, :self.d_model], (activated_state, state_trace, decay_alpha, decay_beta, synchronisation)

class CrossAttentionCTMCell(_CTMCellBase):
    """CTM with cross-attention between synchronisation signal and input tokens."""

    def __init__(self, d_input, d_model, memory_length, dropout, plastic=False, num_tokens=16, num_cross_attn_heads=4, nlm_init_identity=False, plastic_normalize_weights=True, plastic_adaptive_scaling=True, plastic_fixed_scale_factor=0.3):
        super().__init__(d_input, d_model, memory_length, dropout, plastic, plastic_normalize_weights=plastic_normalize_weights, plastic_adaptive_scaling=plastic_adaptive_scaling, plastic_fixed_scale_factor=plastic_fixed_scale_factor)

        # Need two non-overlapping d_model subsets of sync: [:d_model] for output, [d_model:2*d_model] for query
        if not plastic:
            self.sync_repr_size = 2 * d_model
            self.synchronizer = Synchronizer(d_model=d_model, sync_repr_size=2 * d_model)

        self.num_tokens = num_tokens
        self.num_cross_attn_heads = num_cross_attn_heads
        token_dim = d_input // num_tokens

        self.kv_proj = nn.Linear(token_dim, d_model)
        self.q_proj = nn.Linear(d_model, d_model)

        self.nlms = nn.Sequential(SuperLinear(in_dims=memory_length, out_dims=2, N=d_model, dropout=dropout, init_identity=nlm_init_identity), nn.GLU(), Squeeze(-1))
        self.start_trace = nn.Parameter(torch.zeros((d_model, memory_length)))
        self.start_activated_state = nn.Parameter(torch.zeros((d_model)))

    def reset_parameters(self):
        super().reset_parameters()
        self.nlms[0].reset_parameters()
        self.kv_proj.reset_parameters()
        self.q_proj.reset_parameters()
        nn.init.zeros_(self.start_trace)
        nn.init.zeros_(self.start_activated_state)

    def initialize_state(self):
        state_trace = self.start_trace.unsqueeze(0) # Shape: (1, H, T)
        activated_state = self.start_activated_state.unsqueeze(0) # Shape: (1, H)
        decay_alpha, decay_beta, initial_synch = self._init_sync_state(activated_state)
        return (activated_state, state_trace, decay_alpha, decay_beta, initial_synch)

    def forward(self, input, state):
        assert input.ndim == 2, "Input should be of size B x D"
        B = input.shape[0]
        activated_state, state_trace, decay_alpha, decay_beta, synchronisation = self.dropout(state[0]), state[1], state[2], state[3], state[4]
        activated_state, state_trace, decay_alpha, decay_beta, synchronisation = self._expand_state((activated_state, state_trace, decay_alpha, decay_beta, synchronisation), B)

        tokens = input.view(B, self.num_tokens, -1)
        KV = self.kv_proj(tokens) # (B, num_tokens, d_model)
        Q = self.q_proj(synchronisation[:, self.d_model:2 * self.d_model]).unsqueeze(1) # (B, 1, d_model)

        H, d_k = self.num_cross_attn_heads, self.d_model // self.num_cross_attn_heads
        Q = Q.view(B, 1, H, d_k).transpose(1, 2)
        KV = KV.view(B, self.num_tokens, H, d_k).transpose(1, 2)
        observation = F.scaled_dot_product_attention(Q, KV, KV).transpose(1, 2).contiguous().view(B, self.d_model)

        state = self.synapses(torch.concat([observation, activated_state], dim=-1), weights_plastic=synchronisation.view(B, self.d_model, self.d_model) if self.plastic else None)
        state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

        activated_state = self.nlms(state_trace)

        synchronisation, decay_alpha, decay_beta = self.synchronizer(activated_state, decay_alpha, decay_beta)

        return synchronisation[:, :self.d_model], (activated_state, state_trace, decay_alpha, decay_beta, synchronisation)


class ContinuousThoughtMachineCellNoNLMs(_CTMCellBase):
    """CTM variant without Neuron-Level Models (NLMs)."""

    def __init__(self, d_input, d_model, memory_length, dropout, plastic=False, plastic_normalize_weights=True, plastic_adaptive_scaling=True, plastic_fixed_scale_factor=0.3):
        super().__init__(d_input, d_model, memory_length, dropout, plastic, plastic_normalize_weights=plastic_normalize_weights, plastic_adaptive_scaling=plastic_adaptive_scaling, plastic_fixed_scale_factor=plastic_fixed_scale_factor)
        self.start_state = nn.Parameter(torch.zeros((d_model)))

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.zeros_(self.start_state)

    def initialize_state(self):
        state = self.start_state.unsqueeze(0)  # Shape: (1, H)
        decay_alpha, decay_beta, initial_synch = self._init_sync_state(state)
        return (state, decay_alpha, decay_beta, initial_synch)

    def forward(self, input, state):
        assert input.ndim == 2, "Input should be of size B x D"
        B = input.shape[0]
        state, decay_alpha, decay_beta, synchronisation = self.dropout(state[0]), state[1], state[2], state[3]
        state, decay_alpha, decay_beta, synchronisation = self._expand_state((state, decay_alpha, decay_beta, synchronisation), B)

        state = self.synapses(torch.concat([input, state], dim=-1), weights_plastic=synchronisation.view(B, self.d_model, self.d_model) if self.plastic else None)

        synchronisation, decay_alpha, decay_beta = self.synchronizer(state, decay_alpha, decay_beta)

        return synchronisation[:, :self.d_model], (state, decay_alpha, decay_beta, synchronisation)

class ContinuousThoughtMachineHebbianCell(ContinuousThoughtMachineCell):
    """Plastic CTM variant where traditional Hebbian update rules is applied to synapse."""

    def __init__(self, d_input, d_model, memory_length, dropout, plastic=True, nlm_init_identity=False):
        assert plastic==True, "Traditional Hebbian CTM cell requires plastic=True."
        super().__init__(d_input, d_model, memory_length, dropout, plastic, nlm_init_identity=nlm_init_identity)
        self.sync_repr_size = d_model
        self.synchronizer = Synchronizer(d_model=d_model, sync_repr_size=d_model)
        self.synapses = HebbianPlasticLinear(hidden_features=d_model) # Replace PlasticLinear with HebbianPlasticLinear
        self.Hebb_0 = nn.Buffer(torch.zeros(d_model, d_model))
        self.eta = nn.Parameter(torch.full((1,), 0.01))

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.constant_(self.eta, 0.01)

    def initialize_state(self):
        activated_state, state_trace, decay_alpha, decay_beta, initial_synch = super().initialize_state()
        Hebb_0 = self.Hebb_0.unsqueeze(0)  # Shape: (1, d_model, d_model)
        return (activated_state, state_trace, decay_alpha, decay_beta, initial_synch, Hebb_0)

    def forward(self, input, state):
        assert input.ndim == 2, "Input should be of size B x D"
        B = input.shape[0]
        activated_state, state_trace, decay_alpha, decay_beta, synchronisation, Hebb_0 = self.dropout(state[0]), state[1], state[2], state[3], state[4], state[5]
        activated_state, state_trace, decay_alpha, decay_beta, synchronisation, Hebb_0 = self._expand_state((activated_state, state_trace, decay_alpha, decay_beta, synchronisation, Hebb_0), B)

        state = self.synapses(torch.concat([input, activated_state], dim=-1), weights_plastic=Hebb_0)
        state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

        h_0 = state_trace[:,:,-2]
        h_1 = state_trace[:,:,-1]
        Hebb_1 = self.eta * torch.einsum("bi,bj->bij", h_0, h_1) + (1 - self.eta) * Hebb_0  # Hebb_i,j_t = η⋅x_i_t-1 ⊗ x_j_t + (1 - η)⋅Hebb_i,j_t-1

        activated_state = self.nlms(state_trace)

        synchronisation, decay_alpha, decay_beta = self.synchronizer(activated_state, decay_alpha, decay_beta)

        return synchronisation[:, :self.d_model], (activated_state, state_trace, decay_alpha, decay_beta, synchronisation, Hebb_1) # Use synch regardless of plastic
