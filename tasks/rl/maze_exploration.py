from typing import Any

import numpy as np
from gymnasium import Env
from gymnasium.envs.registration import register
from gymnasium.spaces import Box, Discrete
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch import Tensor, nn
import torch
from torch.distributions import Categorical, Distribution
from hydra.core.hydra_config import HydraConfig

from models import construct_rnn
from tasks.rl.ppo import RLTask
from tasks.rl.rl2 import RL2
from tasks.rl.popart import PopArtLayer


# Maze environment, from "Differentiable plasticity: training plastic neural networks with backpropagation"
class Maze(Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

  def __init__(self, render_mode: str | None = None, maze_size: int = 7):
    super().__init__()
    self.grid_size = maze_size
    self.observation_space = Box(low=0, high=1, shape=(9,), dtype=np.float32)  # Observation: flattened 3 x 3 binary grid
    self.action_space = Discrete(4)
    self.action_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # Action mapping: 0 → Up, 1 → Right, 2 → Down, 3 → Left
    self.render_mode, self.cell_size = render_mode, 32  # Size of each grid for rendering

    self.grid = self._create_maze()
    self.empty_locs = np.argwhere(self.grid == 0)

    self.agent_pos, self.goal_pos, self.step_count = None, None, 0

  def _create_maze(self) -> np.array:
    grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
    grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1] = 1, 1, 1, 1  # Borders
    # Place pillars at even-numbered positions (2, 4, 6, ..., grid_size-3)
    pillar_positions = list(range(2, self.grid_size - 1, 2))
    for r in pillar_positions:
      for c in pillar_positions:
        grid[r, c] = 1  # Pillars
    return grid

  def _get_obs(self) -> np.array:
    r, c = self.agent_pos
    window = self.grid[r - 1 : r + 2, c - 1 : c + 2].flatten()  # Extract 3 x 3 window around agent and flatten
    return window.astype(np.float32)

  def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.array, dict[str, np.array]]:
    super().reset(seed=seed, options=options)
    self.step_count = 0
    self.goal_pos = tuple(self.empty_locs[np.random.choice(len(self.empty_locs))])  # Randomise goal position
    self.agent_pos = tuple(self.empty_locs[np.random.choice(len(self.empty_locs))])  # Randomise agent position
    if self.render_mode == "human":
      self._render_frame()
    return self._get_obs(), {}

  def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict[str, np.array]]:
    self.step_count += 1
    reward = 0

    delta = self.action_map[action]
    next_r, next_c = self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1]  # Calculate next (potential) position

    if self.grid[next_r, next_c] == 1:
      reward = -0.1  # Penalise walking into a wall
    else:
      self.agent_pos = (next_r, next_c)
      if self.agent_pos == self.goal_pos:
        reward = 10  # Reward agent for reaching goal
        self.agent_pos = tuple(self.empty_locs[np.random.choice(len(self.empty_locs))])  # Randomise agent position

    terminated, truncated = False, False
    if self.render_mode == "human":
      self._render_frame()
    return self._get_obs(), reward, terminated, truncated, {}

  def _render_frame(self) -> np.array:
    colour_ground, colour_walls, colour_agent, colour_goal = np.array([68, 1, 84]), np.array([55, 90, 127]), np.array([253, 231, 37]), np.array([90, 200, 100])

    img = np.zeros((self.grid_size * self.cell_size, self.grid_size * self.cell_size, 3), dtype=np.uint8)  # Create base canvas
    img[:, :] = colour_ground  # Fill background (ground)
    wall_idxs = np.argwhere(self.grid == 1)
    for r, c in wall_idxs:
      r_start, r_end = r * self.cell_size, (r + 1) * self.cell_size
      c_start, c_end = c * self.cell_size, (c + 1) * self.cell_size
      img[r_start:r_end, c_start:c_end] = colour_walls  # Draw walls
    if self.goal_pos:
      r, c = self.goal_pos
      r_start, r_end = r * self.cell_size, (r + 1) * self.cell_size
      c_start, c_end = c * self.cell_size, (c + 1) * self.cell_size
      img[r_start:r_end, c_start:c_end] = colour_goal  # Draw goal
    if self.agent_pos:
      r, c = self.agent_pos
      r_start, r_end = r * self.cell_size, (r + 1) * self.cell_size
      c_start, c_end = c * self.cell_size, (c + 1) * self.cell_size
      img[r_start:r_end, c_start:c_end] = colour_agent  # Draw agent

    if self.render_mode == "human":
      plt.imshow(img)
      plt.axis("off")
      plt.show(block=False)
      plt.pause(0.1)
      plt.clf()
    return img

  def render(self):
    if self.render_mode == "rgb_array":
      return self._render_frame()


register(id="Maze-v0", entry_point="tasks.rl.maze_exploration:Maze")


class MazeModel(nn.Module):
  def __init__(self, cfg: DictConfig, model_type: str):
    super().__init__()
    self.rnn = construct_rnn(d_input=9 + 4 + 1, d_model=cfg.model.d_model, model_type=model_type, model_kwargs=cfg.model)
    self.actor = nn.Linear(cfg.model.d_model, 4)
    self.critic = PopArtLayer(cfg.model.d_model, 1, beta=cfg.training.popart_step_size)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.orthogonal_(self.actor.weight)
    nn.init.zeros_(self.actor.bias)
    nn.init.orthogonal_(self.critic.weight)
    nn.init.zeros_(self.critic.bias)

  def forward(self, input: Tensor, state: tuple[Tensor, ...]) -> tuple[Distribution, Tensor, Tensor, tuple[Tensor, ...]]:  # noqa: A002
    input = input.unsqueeze(0)  # (B, D) -> (1, B, D) Add time dimension
    hidden, state = self.rnn(input, state)
    hidden = hidden.squeeze(0)  # (1, B, D) -> (B, D) Remove time dimension
    policy = Categorical(logits=self.actor(hidden))
    normalised_value, unnormalised_value = self.critic(hidden)
    return policy, normalised_value.squeeze(), unnormalised_value.squeeze(), state

# Maze exploration task, from "Differentiable plasticity: training plastic neural networks with backpropagation"
class MazeExploration(RLTask):
  def __init__(self, cfg: DictConfig):
    super().__init__(cfg, env="Maze-v0", vectorization_mode="sync", wrappers=(RL2,), max_episode_steps=cfg.task.max_episode_steps, maze_size=cfg.task.maze_size)

  @staticmethod
  def build_model(cfg: DictConfig, model_type: str | None = None) -> nn.Module:
    if model_type is None:
      model_type = HydraConfig.get().runtime.choices.model
    model = MazeModel(cfg, model_type)
    # Materialise lazy modules so the parameter count is accurate (needed for max_parameters)
    pseudo_inputs = torch.zeros(1, 9 + 4 + 1).float()
    model(pseudo_inputs, model.rnn.get_initial_state())
    return model

  def init_lazy_modules(self):
    input_shape = (1, 9 + 4 + 1)
    pseudo_inputs = torch.zeros(input_shape, device=self.device).float()
    self.model(pseudo_inputs, self.model.rnn.get_initial_state())