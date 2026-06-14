import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CURRENT_PATH, "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = 1e-6


class GaussianPolicy(nn.Module):
    def __init__(self, n_features, n_actions, action_space=None):
        super(GaussianPolicy, self).__init__()

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
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
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
    def __init__(self, n_features, n_actions):
        super().__init__()

        self.fc1_1 = nn.Linear(n_features + n_actions, 400)
        self.fc1_2 = nn.Linear(400, 300)
        self.fc1_3 = nn.Linear(300, 1)

        self.fc2_1 = nn.Linear(n_features + n_actions, 400)
        self.fc2_2 = nn.Linear(400, 300)
        self.fc2_3 = nn.Linear(300, 1)

        self.to(DEVICE)

    def forward(self, x, action) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = torch.cat(tensors=[x, action], dim=-1)

        x1 = F.relu(self.fc1_1(x))
        x1 = F.relu(self.fc1_2(x1))
        x1 = self.fc1_3(x1)

        x2 = F.relu(self.fc2_1(x))
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.fc2_3(x2)
        return x1, x2


class SumTree:
    """Binary sum-tree over `capacity` leaves supporting O(log n) priority updates and sampling."""

    def __init__(self, capacity: int):
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self.tree_capacity = tree_capacity
        self.tree = np.zeros(2 * tree_capacity, dtype=np.float64)

    def update(self, idx: int, priority: float) -> None:
        tree_idx = idx + self.tree_capacity
        self.tree[tree_idx] = priority
        tree_idx //= 2
        while tree_idx >= 1:
            self.tree[tree_idx] = self.tree[2 * tree_idx] + self.tree[2 * tree_idx + 1]
            tree_idx //= 2

    def total(self) -> float:
        return self.tree[1]

    def get(self, cumsum: float) -> int:
        idx = 1
        while idx < self.tree_capacity:
            left = 2 * idx
            if cumsum <= self.tree[left]:
                idx = left
            else:
                cumsum -= self.tree[left]
                idx = left + 1
        return idx - self.tree_capacity

    def priority(self, idx: int) -> float:
        return self.tree[idx + self.tree_capacity]


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay (Schaul et al., 2016) for continuous-action SAC.

    Transitions are stored densely (same layout as a uniform replay buffer); a sum-tree over
    priorities = |TD-error|^alpha drives stratified sampling, and importance-sampling weights
    (annealed via beta) correct the resulting bias in the critic loss.
    """

    def __init__(
        self,
        capacity: int,
        n_features: int,
        n_actions: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 1_000_000,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.ptr = 0
        self.num_transitions = 0

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1
        self.max_priority = 1.0

        # Pre-allocate the entire buffer as tensors on DEVICE (assumes DEVICE is cuda)
        self.observations = torch.zeros((capacity, n_features), dtype=torch.float32, device=DEVICE)
        self.actions = torch.zeros((capacity, n_actions), dtype=torch.float32, device=DEVICE)
        self.next_observations = torch.zeros((capacity, n_features), dtype=torch.float32, device=DEVICE)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=DEVICE)
        self.dones = torch.zeros((capacity,), dtype=torch.bool, device=DEVICE)

        self.tree = SumTree(capacity)

    def size(self) -> int:
        return self.num_transitions

    def append(self, observation, action, next_observation, reward, done) -> None:
        self.observations[self.ptr] = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=DEVICE)
        self.next_observations[self.ptr] = torch.as_tensor(next_observation, dtype=torch.float32, device=DEVICE)
        self.rewards[self.ptr, 0] = float(reward)
        self.dones[self.ptr] = bool(done)

        # New transitions get the current max priority so they are sampled at least once
        self.tree.update(self.ptr, self.max_priority ** self.alpha)

        self.ptr = (self.ptr + 1) % self.capacity # circular buffer
        self.num_transitions = min(self.num_transitions + 1, self.capacity)

    def _beta(self) -> float:
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
        total = self.tree.total()
        segment = total / batch_size

        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        for i in range(batch_size):
            cumsum = np.random.uniform(segment * i, segment * (i + 1))
            idx = self.tree.get(cumsum)
            indices[i] = idx
            priorities[i] = self.tree.priority(idx)

        probs = priorities / total
        beta = self._beta()
        weights = (self.num_transitions * probs) ** (-beta)
        weights /= weights.max()
        self.frame += 1

        indices_t = torch.as_tensor(indices, device=DEVICE)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        return (
            self.observations[indices_t],
            self.actions[indices_t],
            self.next_observations[indices_t],
            self.rewards[indices_t],
            self.dones[indices_t],
            indices,
            weights_t,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = np.abs(td_errors) + self.epsilon
        for idx, priority in zip(indices, priorities):
            self.tree.update(int(idx), float(priority) ** self.alpha)
        self.max_priority = max(self.max_priority, float(priorities.max()))
