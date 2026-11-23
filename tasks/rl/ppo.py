import math

import gymnasium as gym
import imageio
import numpy as np
import torch
from gymnasium import Wrapper
from gymnasium.spaces import Discrete, Space
from gymnasium.wrappers.vector import NumpyToTorch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tasks.task import Task
from tasks.utils import get_device


# Tiles a list of images into a square(ish) grid; assumes all frames are the same H x W x C dimensions
def _make_render_grid(frames: list[np.array]) -> np.array:
  n_maps = len(frames)
  H, W, C = frames[0].shape
  grid_cols = math.ceil(math.sqrt(n_maps))  # Calculate grid size (approx square)
  grid_rows = math.ceil(n_maps / grid_cols)
  grid_img = np.zeros((grid_rows * H, grid_cols * W, C), dtype=frames[0].dtype)  # Initialise blank canvas (black background)
  for i, frame in enumerate(frames):
    row, col = i // grid_cols, i % grid_cols
    y_start, x_start = row * H, col * W
    y_end, x_end = y_start + H, x_start + W
    grid_img[y_start:y_end, x_start:x_end] = frame
  return grid_img


class TrajectoryBuffer(Dataset):
  def __init__(self, num_steps: int, num_envs: int, seq_len: int, observation_size: int, action_space: Space, state_shapes: tuple[torch.Size, ...]):
    assert num_steps % seq_len == 0, f"The replay size {num_steps} should be divisible by the training sequence length {seq_len}"
    self.num_steps, self.num_envs, self.seq_len, self.num_sequences_per_env, self.idx = num_steps, num_envs, seq_len, num_steps // seq_len, 0
    self.observations, self.rewards, self.terminateds, self.truncateds = torch.zeros(num_steps, num_envs, observation_size), torch.zeros(num_steps, num_envs), torch.zeros(num_steps, num_envs), torch.zeros(num_steps, num_envs)  # Storage of (o_t, r_t+1, terminated_t+1, truncated_t+1)
    self.actions = torch.zeros(num_steps, num_envs) if isinstance(action_space, Discrete) else torch.zeros(num_steps, num_envs, action_space.shape[0])  # Storage of (a_t,)
    self.log_prob_actions, self.values, self.bootstrap_values = torch.zeros(num_steps, num_envs), torch.zeros(num_steps, num_envs), torch.zeros(num_steps, num_envs)  # Storage of (log(π(a_t|o_t)), V_t, V_t+1)
    self.returns, self.advantages = torch.zeros(num_steps, num_envs), torch.zeros(num_steps, num_envs)  # Storage of (R^_t, A^_t)
    self.states = [torch.zeros((num_steps, num_envs) + s_shape) for s_shape in state_shapes]  # Storage of states
    self.valid = torch.zeros(num_steps, num_envs, dtype=torch.bool)  # Storage of valid transitions (not reset steps)

  def store(self, observation: Tensor, action: Tensor, reward: Tensor, terminated: Tensor, truncated: Tensor, log_prob_action: Tensor, value: Tensor, state: tuple[Tensor, ...], bootstrap_value: Tensor, valid: Tensor):
    assert self.idx < self.num_steps, "Trajectory buffer is full"
    self.observations[self.idx], self.actions[self.idx], self.rewards[self.idx], self.terminateds[self.idx], self.truncateds[self.idx] = observation, action, reward, terminated, truncated  # Store (o_t, a_t, r_t+1, terminated_t+1, truncated_t+1)
    self.log_prob_actions[self.idx], self.values[self.idx], self.bootstrap_values[self.idx] = log_prob_action, value, bootstrap_value  # Store (log(π(a_t|o_t)), V_t, V_t+1)
    for i, s in enumerate(state):
      self.states[i][self.idx] = s.detach()  # Store states
    self.valid[self.idx] = valid
    self.idx += 1

  def reset(self):
    self.idx = 0

  def compute_gae(self, next_value: Tensor, discount: float, trace_decay: float):
    advantage = torch.zeros(self.num_envs)
    for t in reversed(range(self.num_steps)):
      next_value = torch.where(self.truncateds[t].bool(), self.bootstrap_values[t], next_value)  # Use bootstrap value if truncated, otherwise next value
      td_error = self.rewards[t] + (1 - self.terminateds[t]) * discount * next_value - self.values[t]  # TD-error δ = r_t + (1 - d)⋅γ⋅V(s_t+1) - V(s_t)
      advantage = td_error + (1 - torch.max(self.terminateds[t], self.truncateds[t])) * discount * trace_decay * advantage  # Advantage estimate A^_t = δ + (1 - d)⋅γ⋅λ⋅A^_t+1
      self.advantages[t] = advantage
      next_value = self.values[t]
    self.returns = self.advantages + self.values  # TD(λ) return estimate

  def __len__(self):
    return self.num_envs * self.num_sequences_per_env

  def __getitem__(self, idx: int) -> dict[str, Tensor | tuple[Tensor]]:
    env_idx, seq_idx = idx // self.num_sequences_per_env, idx % self.num_sequences_per_env
    seq_start_idx, seq_end_idx = seq_idx * self.seq_len, (seq_idx + 1) * self.seq_len
    observations, actions, dones = self.observations[seq_start_idx:seq_end_idx, env_idx], self.actions[seq_start_idx:seq_end_idx, env_idx], torch.max(self.terminateds[seq_start_idx:seq_end_idx, env_idx], self.truncateds[seq_start_idx:seq_end_idx, env_idx])
    log_prob_actions, values, returns, advantages = self.log_prob_actions[seq_start_idx:seq_end_idx, env_idx], self.values[seq_start_idx:seq_end_idx, env_idx], self.returns[seq_start_idx:seq_end_idx, env_idx], self.advantages[seq_start_idx:seq_end_idx, env_idx]
    state = tuple(s[seq_start_idx, env_idx] for s in self.states)
    valid = self.valid[seq_start_idx:seq_end_idx, env_idx]
    return {"observations": observations, "actions": actions, "dones": dones, "log_prob_actions": log_prob_actions, "values": values, "returns": returns, "advantages": advantages, "state": state, "valid": valid}

  @staticmethod
  def collate_fn(batch: list[dict[str, Tensor | tuple[Tensor]]]) -> dict[str, Tensor | tuple[Tensor]]:
    transitions = {}
    for key in ["observations", "actions", "dones", "log_prob_actions", "values", "returns", "advantages", "valid"]:
      transitions[key] = torch.stack([item[key] for item in batch], dim=1)  # Stack transition data as T x B (x D)
    batch_states = tuple(zip(*[item["state"] for item in batch], strict=True))
    transitions["state"] = tuple(torch.stack(list(state_component), dim=0) for state_component in batch_states)  # Stack states as S x B x D
    return transitions


class RLTask(Task):
  def __init__(self, cfg: DictConfig, env: str, eval_env: str | None = None, vectorization_mode: str | None = None, wrappers: tuple[Wrapper] | None = None, eval_kwargs: dict | None = None, **kwargs):
    super().__init__()
    self.device, self.num_envs, self.replay_size, self.seq_len, self.batch_size, self.evaluation_episodes, self.evaluation_interval, self.render_interval = torch.device(get_device(cfg)), cfg.training.num_envs, cfg.training.replay_size, cfg.training.seq_len, cfg.training.batch_size, cfg.evaluation.episodes, cfg.evaluation.evaluation_interval, cfg.evaluation.render_interval
    self.render_envs = cfg.evaluation.get("render_envs", 1)
    default_render_frames = getattr(getattr(cfg, "task", None), "max_episode_steps", 1000)
    self.render_max_frames = cfg.evaluation.get("render_max_frames", default_render_frames)
    self.discount, self.trace_decay, self.ppo_epochs, self.ppo_clip_range, self.vf_coeff, self.entropy_coeff, self.kl_threshold, self.max_grad_norm = cfg.training.discount, cfg.training.trace_decay, cfg.training.ppo_epochs, cfg.training.ppo_clip_range, cfg.training.vf_coeff, cfg.training.entropy_coeff, cfg.training.kl_threshold, cfg.training.max_grad_norm
    self.env = NumpyToTorch(gym.make_vec(env, num_envs=cfg.training.num_envs, vectorization_mode=vectorization_mode, wrappers=wrappers, **kwargs))
    eval_kwargs_merged = {**kwargs, **(eval_kwargs or {})}
    self.eval_env = NumpyToTorch(gym.make_vec(eval_env if eval_env is not None else env, num_envs=cfg.training.num_envs, vectorization_mode=vectorization_mode, wrappers=wrappers, render_mode="rgb_array", **eval_kwargs_merged))
    self.model = self.build_model(cfg).to(self.device)
    self.init_lazy_modules()
    print(f"Total Parameter Count: {self.get_parameter_count()}")
    self.optimizer = AdamW(self.model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)  # No-op scheduler for compatibility
    self.trajectories = TrajectoryBuffer(num_steps=cfg.training.replay_size, num_envs=cfg.training.num_envs, seq_len=cfg.training.seq_len, observation_size=self.env.single_observation_space.shape[0], action_space=self.env.single_action_space, state_shapes=tuple(s.shape[1:] for s in self.model.rnn.get_initial_state()))

  def build_model(self, cfg: DictConfig) -> nn.Module:
    raise NotImplementedError

  def mask_state(self, state: tuple[Tensor, ...], mask: Tensor) -> Tensor:
      initial_state = self.model.rnn.get_initial_state()
      return tuple(mask.view(mask.shape[0], *[1] * (s.ndim - 1)) * s + (1 - mask.view(mask.shape[0], *[1] * (s.ndim - 1))) * initial_s for s, initial_s in zip(state, initial_state, strict=True))

  def train(self, _epoch: int) -> dict[str, float]:
    self.model.train()

    # Collect trajectories by running policy in the environment
    self.trajectories.reset()
    with torch.no_grad():
      autoreset, current_return, train_returns = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), torch.zeros(self.num_envs, device=self.device), []
      (observation, _), state = self.env.reset(), tuple(s.expand(self.num_envs, *[-1] * (s.ndim - 1)) for s in self.model.rnn.get_initial_state())
      observation = observation.to(self.device)
      with tqdm(range(self.replay_size), leave=False, desc="Collecting") as pbar:
        for _ in pbar:
          policy, _, value, next_state = self.model(observation, state) # Collect unnormalised value
          action = policy.sample()
          next_observation, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
          next_observation, reward = next_observation.to(self.device), reward.to(dtype=torch.float32, device=self.device)
          current_return += reward * (~autoreset).float()  # Store rewards for real steps but not reset steps (that overlap episode boundaries (o_final -> o_start))

          bootstrap_value = self.model(next_observation, next_state)[2].detach()  # Calculate and store V(s_terminal) for truncated steps (use unnormalised value)
          self.trajectories.store(observation.detach(), action.detach(), reward, terminated, truncated, policy.log_prob(action).detach(), value.detach(), state, bootstrap_value, ~autoreset)

          dones = torch.logical_or(terminated, truncated)
          observation, state = next_observation, self.mask_state(next_state, 1 - autoreset.float())  # If a standard step is finished (terminated), don't reset yet; if a reset step is finished (autoreset), reset
          if dones.any():
            train_returns.extend(current_return[dones].detach().tolist())
            current_return[dones] = 0
            if len(train_returns) > 0:
              pbar.set_postfix({"Return": f"{sum(train_returns) / len(train_returns):.2f}"})
          autoreset = dones.to(self.device)  # Update autoreset mask for next step

      # Compute generalised advantage estimate
      self.trajectories.compute_gae(self.model(observation, state)[2].detach().cpu(), self.discount, self.trace_decay)

      # Update PopArt statistics once per rollout using all valid returns
      valid_returns = self.trajectories.returns[self.trajectories.valid].to(self.device)
      self.model.critic.update_stats(valid_returns)

    dataloader = DataLoader(self.trajectories, batch_size=self.batch_size, shuffle=True, collate_fn=TrajectoryBuffer.collate_fn)
    objectives = []

    # Compute explained variance before any updates
    with torch.no_grad():
        valid_mask = self.trajectories.valid.flatten()
        all_returns = self.trajectories.returns.flatten()[valid_mask]
        all_values = self.trajectories.values.flatten()[valid_mask]
        explained_var = 1 - (all_returns - all_values).var() / (all_returns.var() + 1e-8)
        value_mean = all_values.mean()
        value_std = all_values.std()
        returns_mean = all_returns.mean()
        returns_std = all_returns.std()

    train_ret_desc = f" (train_ret: {sum(train_returns) / len(train_returns):.2f})" if len(train_returns) > 0 else ""
    with tqdm(range(self.ppo_epochs), leave=False, desc=f"Training{train_ret_desc}") as pbar_train:
      for _ in pbar_train:
        for transitions in dataloader:
          observations, actions, dones, old_log_prob_actions, old_values, returns, advantages, state, valid = [transitions[key] for key in ["observations", "actions", "dones", "log_prob_actions", "values", "returns", "advantages", "state", "valid"]]
          observations, actions, dones, old_log_prob_actions, old_values, returns, advantages, state, valid = observations.to(self.device), actions.to(self.device), dones.to(self.device), old_log_prob_actions.to(self.device), old_values.to(self.device), returns.to(self.device), advantages.to(self.device), tuple(s.to(self.device) for s in state), valid.to(self.device)

          # Process each action step by step
          log_prob_actions, normalized_values, entropies = [], [], []
          for t in range(observations.shape[0]):
            observation = observations[t]
            action = actions[t].long() if isinstance(self.env.single_action_space, Discrete) else actions[t]
            policy, normalised_value, _, state = self.model(observation, state)
            state = self.mask_state(state, valid[t].float())  # Match collection: reset on autoreset steps, not done steps
            log_prob_actions.append(policy.log_prob(action))
            normalized_values.append(normalised_value.squeeze())
            entropies.append(policy.entropy() if isinstance(policy, Categorical) else policy.base_dist.entropy())

          log_prob_actions = torch.stack(log_prob_actions, dim=0)  # (T, B)
          normalized_values = torch.stack(normalized_values, dim=0)  # (T, B)
          entropies = torch.stack(entropies, dim=0)  # (T, B)

          # PPO algorithm, from "Proximal Policy Optimization Algorithms"
          log_prob_ratio = log_prob_actions - old_log_prob_actions
          prob_ratio = log_prob_ratio.exp()  # Probability ratio r_t = π(a_t|s_t) / π_old(a_t|s_t)
          with torch.no_grad():
            approx_kl = (prob_ratio - 1) - log_prob_ratio  # KL[q,p] estimate = (r − 1) − log(r)
            clip_frac = ((prob_ratio < 1 - self.ppo_clip_range) | (prob_ratio > 1 + self.ppo_clip_range)).float()  # Fraction of probability ratios clipped
            # Normalise advantage estimates using only valid timesteps
            adv_valid = advantages[valid]
            adv_mean = adv_valid.mean()
            adv_std = adv_valid.std()
          advantages = (advantages - adv_mean) / (adv_std + 1e-8)
          policy_obj = torch.min(prob_ratio * advantages, torch.clamp(prob_ratio, min=1 - self.ppo_clip_range, max=1 + self.ppo_clip_range) * advantages)  # L_clip = min(r_t⋅A^_t, clip(r_t, 1 − ε, 1 + ε)⋅A^_t)

          normalised_old_values, normalised_returns = self.model.critic.normalise_target(old_values), self.model.critic.normalise_target(returns)  # Normalise old values and returns to new range
          clipped_normalised_values = normalised_old_values + (normalized_values - normalised_old_values).clamp(-self.ppo_clip_range, self.ppo_clip_range)
          clipped_value_loss = torch.max((normalized_values - normalised_returns) ** 2, (clipped_normalised_values - normalised_returns) ** 2)  # L_vf_clip = max((V(s_t) − V_targ_t)^2, (V_clip(s_t) − V_targ_t)^2)

          policy_obj, clipped_value_loss, entropies, approx_kl, clip_frac = policy_obj[valid].mean(), clipped_value_loss[valid].mean(), entropies[valid].mean(), approx_kl[valid].mean(), clip_frac[valid].mean()  # Extract valid components and average
          if self.kl_threshold > 0 and approx_kl.item() > self.kl_threshold:
            break  # Perform early stopping based on KL-divergence (KLE-Stop, from "An Empirical Investigation of Early Stopping Optimizations in Proximal Policy Optimization")
          loss = -policy_obj + self.vf_coeff * clipped_value_loss + -self.entropy_coeff * entropies  # L_clip - c_1⋅L_vf_clip + c_2⋅S[π](s_t)
          self.optimizer.zero_grad()
          loss.backward()
          grad_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
          self.optimizer.step()
          objectives.append((policy_obj.item(), clipped_value_loss.item(), entropies.item(), approx_kl.item(), clip_frac.item(), grad_norm.item()))
    objectives = [sum(o) / len(o) for o in zip(*objectives, strict=True)]  # Calculate average objective values over training
    metrics = {
        "policy_obj": objectives[0],
        "value_loss": objectives[1],
        "entropy": objectives[2],
        "approx_kl": objectives[3],
        "clip_frac": objectives[4],
        "grad_norm": objectives[5],
        "explained_variance": explained_var.item(),
        "value_mean": value_mean.item(),
        "value_std": value_std.item(),
        "returns_mean": returns_mean.item(),
        "returns_std": returns_std.item(),
    } if len(objectives) > 0 else {}

    if len(train_returns) > 0:
      metrics["return"] = sum(train_returns) / len(train_returns)
    return metrics

  def validate(self, epoch: int) -> dict[str, float]:
    if epoch % self.evaluation_interval != 0:
      return {}

    self.model.eval()

    rendering_enabled = self.render_interval and self.render_interval > 0 and epoch % self.render_interval == 0
    autoreset, episode_counts, current_return, eval_returns, frames = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), torch.zeros(self.num_envs, dtype=torch.long), torch.zeros(self.num_envs, device=self.device), [], []
    with tqdm(total=self.num_envs * self.evaluation_episodes, leave=False) as pbar:
      with torch.inference_mode():
        (observation, _), state = self.eval_env.reset(), tuple(s.expand(self.num_envs, *[-1] * (s.ndim - 1)) for s in self.model.rnn.get_initial_state())
        observation = observation.to(self.device)
        if rendering_enabled:
          raw_frames = self.eval_env.render()
          frames.append(_make_render_grid(raw_frames[: self.render_envs]))
        while torch.min(episode_counts) < self.evaluation_episodes:
          policy, _, _, next_state = self.model(observation, state)
          action = torch.argmax(policy.logits, dim=-1) if isinstance(policy, Categorical) else policy.mean  # Use a greedy deterministic policy
          next_observation, reward, terminated, truncated, _ = self.eval_env.step(action.cpu().numpy())
          next_observation, reward = next_observation.to(self.device), reward.to(dtype=torch.float32, device=self.device)
          current_return += reward * (~autoreset).float()  # Store rewards for real steps but not reset steps (that overlap episode boundaries (o_final -> o_start))
          if rendering_enabled and len(frames) < self.render_max_frames:
            raw_frames = self.eval_env.render()
            frames.append(_make_render_grid(raw_frames[: self.render_envs]))

          dones = torch.logical_or(terminated, truncated)
          observation, state = next_observation, self.mask_state(next_state, 1 - autoreset.float())
          if dones.any():
            valid_dones = torch.logical_and(dones, episode_counts < self.evaluation_episodes)
            if valid_dones.any():
              eval_returns.extend(current_return[valid_dones].detach().tolist())
              episode_counts[valid_dones] += 1
              pbar.update(valid_dones.sum().item())
            current_return[dones] = 0
          autoreset = dones.to(self.device)  # Update autoreset mask for next step
    if rendering_enabled and len(frames) > 0:
      imageio.mimsave(f"eval_{epoch}.mp4", frames, fps=30, codec="libx264")
    return {"return": sum(eval_returns) / len(eval_returns)}

  def calculate_performance(self, metrics: dict[str, float]) -> float:
    return sum(pair[1] for pair in metrics["eval_return"])  # Calculative cumulative return

  def perform_diagnostics(self):
    pass

  def __del__(self):
    """Cleanup vectorized environments to prevent resource leaks"""
    if hasattr(self, 'env'):
      self.env.close()
    if hasattr(self, 'eval_env'):
      self.eval_env.close()
