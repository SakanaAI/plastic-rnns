import torch
from torch import nn, Tensor
from torch.nn import functional as F


# Layer to implement the Pop-Art algorithm (special case of scalar scale and shift), from "Learning values across many orders of magnitude"
class PopArtLayer(nn.Module):
  def __init__(self, input_size: int, output_size: int, beta: float = 1e-4, epsilon: float = 1e-5):
    super().__init__()
    self.output_size, self.beta, self.epsilon, self.t = output_size, beta, epsilon, 0
    self.weight, self.bias = nn.Parameter(torch.empty(output_size, input_size)), nn.Parameter(torch.empty(output_size))
    self.mu, self.sigma, self.nu = nn.Buffer(torch.zeros(output_size)), nn.Buffer(torch.ones(output_size)), nn.Buffer(torch.ones(output_size))  # Statistics buffers
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.orthogonal_(self.weight)
    nn.init.zeros_(self.bias)

  def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:  # noqa: A002
    output = F.linear(input, self.weight, self.bias)
    return output, self.sigma * output + self.mu  # Return normalised and unnormalised values g(x) and f(x) = Σ⋅g(x) + µ

  def update_stats(self, targets: Tensor):
    targets = targets.view(-1, self.output_size)  # Flatten targets
    self.t += 1
    # POP: Update scale Σ and shift µ such that Σ⌃-1⋅(Y - µ) is appropriately normalised (zero mean and unit variance)
    beta_t = self.beta / (1 - (1 - self.beta)**self.t)  # Calculate time-dependent β: β_t = β(1 - (1 - β)^t)^-1
    sigma_old, mu_old = self.sigma.clone(), self.mu.clone()
    self.mu.lerp_(targets.mean(dim=0), beta_t)  # µ_t = (1 - β_t)µ_t-1 + β_t⋅Y
    self.nu.lerp_((targets**2).mean(dim=0), beta_t) # ν_t = (1 - β_t)ν_t-1 + β_t⋅Y^2
    self.sigma.copy_(torch.sqrt(self.nu - self.mu**2).clamp(min=self.epsilon))  # σ_t^2 = ν_t - µ_t^2; lower bound with ε > 0 to avoid issues with numerical precision
    # ART: Preserve the outputs of the unnormalised function when we change the scale and shift; rescale W and b
    with torch.no_grad():  # In-place updates technically invalidate optimiser states but changes are small enough
      self.weight.mul_((sigma_old / self.sigma).view(-1, 1))  # W_new = (σ/σ_new)W
      self.bias.copy_((sigma_old * self.bias + mu_old - self.mu) / self.sigma)  # b_new = (σb + µ - µ_new) / σ_new

  def normalise_target(self, targets: Tensor) -> Tensor:
    return (targets - self.mu) / self.sigma  # Y~_t = Σ_t^-1⋅(Y_t -  µ_t)