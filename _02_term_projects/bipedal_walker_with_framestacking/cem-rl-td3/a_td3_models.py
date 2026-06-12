import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CURRENT_PATH, "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _flatten_obs(state, obs_ndim: int):
    """Flatten trailing obs dims into the feature dim, keeping a possible batch dim.

    - stacked obs (obs_ndim == 2): (B, 4, 24) -> (B, 96); (4, 24) -> (96,)
    - unstacked obs (obs_ndim == 1): no-op, keeps (B, 24) / (24,)
    """
    if isinstance(state, np.ndarray):
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
    return state.flatten(start_dim=-obs_ndim)


class Actor(nn.Module):
    def __init__(self, n_features: int, n_actions: int, hidden_dim=(400, 300),
                 obs_ndim: int = 2, exploration_noise: float = 0.1):
        super().__init__()
        self.n_actions = n_actions
        self.obs_ndim = obs_ndim
        self.exploration_noise = exploration_noise

        self.fc1 = nn.Linear(n_features, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.out = nn.Linear(hidden_dim[1], n_actions)
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_obs(x, self.obs_ndim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu_v = torch.tanh(self.out(x))  # BipedalWalker actions are already in [-1, 1]
        return mu_v

    def get_action(self, x: torch.Tensor, exploration: bool = True) -> np.ndarray:
        mu_v = self.forward(x)
        action = mu_v.detach().cpu().numpy()

        if exploration:
            noises = np.random.normal(size=self.n_actions, loc=0.0, scale=self.exploration_noise)
            action = action + noises

        return np.clip(action, a_min=-1.0, a_max=1.0)


class TwinQCritic(nn.Module):
    """TD3 twin critic: two independent Q-networks to reduce overestimation bias."""

    def __init__(self, n_features: int, n_actions: int, hidden_dim=(400, 300), obs_ndim: int = 2):
        super().__init__()
        self.obs_ndim = obs_ndim

        self.q1_fc1 = nn.Linear(n_features, hidden_dim[0])
        self.q1_fc2 = nn.Linear(hidden_dim[0] + n_actions, hidden_dim[1])
        self.q1_fc3 = nn.Linear(hidden_dim[1], 1)

        self.q2_fc1 = nn.Linear(n_features, hidden_dim[0])
        self.q2_fc2 = nn.Linear(hidden_dim[0] + n_actions, hidden_dim[1])
        self.q2_fc3 = nn.Linear(hidden_dim[1], 1)

        self.to(DEVICE)

    def forward(self, x, action) -> tuple[torch.Tensor, torch.Tensor]:
        x = _flatten_obs(x, self.obs_ndim)

        q1 = F.relu(self.q1_fc1(x))
        q1 = torch.cat([q1, action], dim=-1)
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        q2 = F.relu(self.q2_fc1(x))
        q2 = torch.cat([q2, action], dim=-1)
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)
        return q1, q2

    def q1_value(self, x, action) -> torch.Tensor:
        x = _flatten_obs(x, self.obs_ndim)
        q1 = F.relu(self.q1_fc1(x))
        q1 = torch.cat([q1, action], dim=-1)
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        return q1


class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape: tuple, n_actions: int):
        self.capacity = capacity
        self.ptr = 0
        self.num_transitions = 0

        self.observations = torch.zeros((capacity, *observation_shape), dtype=torch.float32, device=DEVICE)
        self.actions = torch.zeros((capacity, n_actions), dtype=torch.float32, device=DEVICE)
        self.next_observations = torch.zeros((capacity, *observation_shape), dtype=torch.float32, device=DEVICE)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=DEVICE)
        self.dones = torch.zeros((capacity,), dtype=torch.bool, device=DEVICE)

    def size(self) -> int:
        return self.num_transitions

    def append(self, observation, action, next_observation, reward, done) -> None:
        self.observations[self.ptr] = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=DEVICE)
        self.next_observations[self.ptr] = torch.as_tensor(next_observation, dtype=torch.float32, device=DEVICE)
        self.rewards[self.ptr, 0] = float(reward)
        self.dones[self.ptr] = bool(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.num_transitions = min(self.num_transitions + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, self.num_transitions, (batch_size,), device=DEVICE)
        return (
            self.observations[indices],
            self.actions[indices],
            self.next_observations[indices],
            self.rewards[indices],
            self.dones[indices],
        )
