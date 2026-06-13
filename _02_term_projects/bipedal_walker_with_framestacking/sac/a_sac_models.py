import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, TransformedDistribution, TanhTransform

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CURRENT_PATH, "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = 1e-6


def _flatten_obs(state, obs_ndim: int):
    """Flatten the trailing `obs_ndim` dims into the feature dim while keeping a possible batch dim.

    - stacked obs (obs_ndim == 2): (B, 4, 24) -> (B, 96); (4, 24) -> (96,)
    - unstacked obs (obs_ndim == 1): no-op, keeps (B, 24) / (24,)
    """
    if isinstance(state, np.ndarray):
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
    return state.flatten(start_dim=-obs_ndim)


class GaussianPolicy(nn.Module):
    def __init__(self, n_features, n_actions, action_space=None, obs_ndim: int = 2):
        super(GaussianPolicy, self).__init__()

        self.obs_ndim = obs_ndim

        self.linear1 = nn.Linear(n_features, 400)
        self.linear2 = nn.Linear(400, 300)

        self.mean_linear = nn.Linear(300, n_actions)
        self.log_std_linear = nn.Linear(300, n_actions)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)
            print("action_space.high: {0}, action_space.low: {1}".format(
                action_space.high, action_space.low
            ))
            print("action_scale: {0}, self.action_bias: {1}".format(
                self.action_scale, self.action_bias
            ))
        self.to(DEVICE)

    def forward(self, state):
        state = _flatten_obs(state, self.obs_ndim)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def get_action(self, state, exploration: bool = True):
        if exploration:
            action, _, _, _ = self.sample(state)
        else:
            _, _, action, _ = self.sample(state)
        return action.detach().cpu().numpy()

    def sample(self, state, reparameterization_trick=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        if reparameterization_trick:
            x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        else:
            x_t = dist.sample()

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = dist.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        entropy = dist.entropy().mean()

        return action, log_prob, mean, entropy

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class SoftQNetwork(nn.Module):
    def __init__(self, n_features, n_actions, obs_ndim: int = 2):
        super().__init__()
        self.obs_ndim = obs_ndim

        self.fc1_1 = nn.Linear(n_features + n_actions, 400)
        self.fc1_2 = nn.Linear(400, 300)
        self.fc1_3 = nn.Linear(300, 1)

        self.fc2_1 = nn.Linear(n_features + n_actions, 400)
        self.fc2_2 = nn.Linear(400, 300)
        self.fc2_3 = nn.Linear(300, 1)

        self.to(DEVICE)

    def forward(self, x, action) -> torch.Tensor:
        x = _flatten_obs(x, self.obs_ndim)
        x = torch.cat(tensors=[x, action], dim=-1)

        x1 = F.relu(self.fc1_1(x))
        x1 = F.relu(self.fc1_2(x1))
        x1 = self.fc1_3(x1)

        x2 = F.relu(self.fc2_1(x))
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.fc2_3(x2)
        return x1, x2


class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape: tuple, n_actions: int):
        self.capacity = capacity
        self.ptr = 0
        self.num_transitions = 0

        # Pre-allocate the entire buffer as tensors on DEVICE (assumes DEVICE is cuda)
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
