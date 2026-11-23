import numpy as np
from gymnasium import Env, Wrapper
from gymnasium.spaces import Box, Discrete


# Wrapper for RL^2 algorithm, from "RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning"/"Learning to reinforcement learn"
class RL2(Wrapper):
  def __init__(self, env: Env):
    super().__init__(env)
    assert isinstance(env.action_space, (Box, Discrete)), "RL2Wrapper only supports Discrete or Box action spaces"
    assert isinstance(env.observation_space, Box) and len(env.observation_space.shape) == 1, "Base environment must have a 1D Box observation space"  # Check observation space (assuming 1D Box for simplicity; generic flattening is complex)
    # Create new observation space
    obs_low, obs_high = env.observation_space.low, env.observation_space.high
    if isinstance(env.action_space, Discrete):
      self.is_discrete_action, self.action_dim = True, env.action_space.n
      act_low, act_high = np.zeros(self.action_dim, dtype=np.float32), np.ones(self.action_dim, dtype=np.float32)
    else:
      self.is_discrete_action, self.action_dim = False, env.action_space.shape[0]
      act_low, act_high = env.action_space.low.flatten(), env.action_space.high.flatten()
    rew_low, rew_high = np.array([-np.inf], dtype=np.float32), np.array([np.inf], dtype=np.float32)
    self.observation_space = Box(low=np.concatenate([obs_low, act_low, rew_low]), high=np.concatenate([obs_high, act_high, rew_high]), dtype=np.float32)
    # Storage for previous step info
    self.prev_action, self.prev_reward = np.zeros(self.action_dim), 0

  def reset(self, **kwargs) -> tuple[np.array, dict[str, np.array]]:
    obs, info = self.env.reset(**kwargs)
    # Reset memory
    self.prev_action, self.prev_reward = np.zeros(self.action_dim), 0
    return self._get_augmented_obs(obs), info

  def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict[str, np.array]]:
    obs, reward, terminated, truncated, info = self.env.step(action)
    # Create augmented observation using the action/reward from this step (which become the previous action/reward for the next step)
    if self.is_discrete_action:
      self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
      self.prev_action[action] = 1
    else:
      self.prev_action = action
    self.prev_reward = float(reward)
    aug_obs = self._get_augmented_obs(obs)
    return aug_obs, reward, terminated, truncated, info

  def _get_augmented_obs(self, obs: np.array) -> np.array:
    return np.concatenate([obs, self.prev_action, [self.prev_reward]]).astype(np.float32)
