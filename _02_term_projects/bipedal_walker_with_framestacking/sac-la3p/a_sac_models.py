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


class SumTree:
    """Vectorized sum-tree (array-of-levels) supporting batched priority updates and sampling.

    Mirrors the `SumTree` used in the reference LA3P implementation
    (https://github.com/baturaysaglam/actor-prioritized-exp-replay): `self.levels[0]` is the
    root (total priority), `self.levels[-1]` holds the per-transition leaf priorities.
    """

    def __init__(self, capacity: int):
        self.levels = [np.zeros(1)]
        level_size = 1
        while level_size < capacity:
            level_size *= 2
            self.levels.append(np.zeros(level_size))

    def total(self) -> float:
        return self.levels[0][0]

    def leaves(self, indices: np.ndarray) -> np.ndarray:
        return self.levels[-1][indices]

    def sample(self, batch_size: int) -> np.ndarray:
        value = np.random.uniform(0, self.levels[0][0], size=batch_size)
        indices = np.zeros(batch_size, dtype=np.int64)

        for nodes in self.levels[1:]:
            indices *= 2
            left_sum = nodes[indices]

            is_greater = np.greater(value, left_sum)
            # If value > left_sum -> go right (+1), else go left (+0)
            indices += is_greater
            # If we go right, subtract the left subtree's sum from the remaining value
            value -= left_sum * is_greater

        return indices

    def set(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        indices, unique_idx = np.unique(indices, return_index=True)
        priority_diff = priorities[unique_idx] - self.levels[-1][indices]

        for nodes in self.levels[::-1]:
            np.add.at(nodes, indices, priority_diff)
            indices = indices // 2


class LA3PReplayBuffer:
    """Actor-Prioritized Experience Replay buffer for SAC + LA3P
    (Saglam et al., "Actor Prioritized Experience Replay",
    https://github.com/baturaysaglam/actor-prioritized-exp-replay).

    Maintains two sum-trees over the same dense storage:

    - `critic_tree`: priorities = `max(|TD-error_1|, |TD-error_2|, min_priority) ** alpha`,
      used by `sample_critic()` for the critic's prioritized batch.
    - `actor_tree`: priorities = `critic_tree.total() / (critic_priority + 1e-6)`, i.e. the
      *reverse* of the critic priority, used by `sample_actor()` so the actor is updated more
      often on transitions the critic already fits well ("easy"/well-understood transitions).

    `sample_uniform()` provides plain uniform sampling for the uniform-batch critic (PAL loss)
    and actor updates.

    Note: the reference implementation rebuilds `actor_tree` from scratch (over every stored
    transition) on every `sample_actor()` call. That is O(buffer size) per training step and,
    measured here, takes ~0.1s at 1M transitions -- too slow to call every step. Instead,
    `actor_tree` is updated incrementally (O(batch size)) alongside `critic_tree`, using the
    current `critic_tree.total()` at update time. This preserves the "prioritize transitions
    the critic already fits well" behavior at a tractable cost, at the price of the reversed
    priorities for *other* transitions drifting slightly stale between their own updates.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple,
        n_actions: int,
        alpha: float = 0.4,
        min_priority: float = 1.0,
    ):
        self.capacity = capacity
        self.ptr = 0
        self.num_transitions = 0

        self.alpha = alpha
        self.min_priority = min_priority
        self.max_priority = min_priority

        # Pre-allocate the entire buffer as tensors on DEVICE (assumes DEVICE is cuda)
        self.observations = torch.zeros((capacity, *observation_shape), dtype=torch.float32, device=DEVICE)
        self.actions = torch.zeros((capacity, n_actions), dtype=torch.float32, device=DEVICE)
        self.next_observations = torch.zeros((capacity, *observation_shape), dtype=torch.float32, device=DEVICE)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=DEVICE)
        self.dones = torch.zeros((capacity,), dtype=torch.bool, device=DEVICE)

        self.critic_tree = SumTree(capacity)
        self.actor_tree = SumTree(capacity)

    def size(self) -> int:
        return self.num_transitions

    def append(self, observation, action, next_observation, reward, done) -> None:
        self.observations[self.ptr] = torch.as_tensor(observation, dtype=torch.float32, device=DEVICE)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=DEVICE)
        self.next_observations[self.ptr] = torch.as_tensor(next_observation, dtype=torch.float32, device=DEVICE)
        self.rewards[self.ptr, 0] = float(reward)
        self.dones[self.ptr] = bool(done)

        # New transitions get the current max priority so they are sampled at least once
        idx = np.array([self.ptr])
        self.critic_tree.set(idx, np.array([self.max_priority]))
        top_value = self.critic_tree.total()
        self.actor_tree.set(idx, np.array([top_value / (self.max_priority + 1e-6)]))

        self.ptr = (self.ptr + 1) % self.capacity
        self.num_transitions = min(self.num_transitions + 1, self.capacity)

    def _gather(self, indices_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.observations[indices_t],
            self.actions[indices_t],
            self.next_observations[indices_t],
            self.rewards[indices_t],
            self.dones[indices_t],
        )

    def sample_uniform(self, batch_size: int):
        """Uniform sampling, used for the uniform-batch critic (PAL loss) and actor update."""
        indices = torch.randint(0, self.num_transitions, (batch_size,), device=DEVICE)
        return (*self._gather(indices), indices.cpu().numpy())

    def sample_critic(self, batch_size: int):
        """Critic-prioritized sampling. Returns indices for `update_priority_critic`."""
        indices = self.critic_tree.sample(batch_size)
        indices_t = torch.as_tensor(indices, device=DEVICE)
        return (*self._gather(indices_t), indices)

    def sample_actor(self, batch_size: int):
        """Actor-prioritized sampling (reverse of critic priority -- "easy" transitions)."""
        indices = self.actor_tree.sample(batch_size)
        indices_t = torch.as_tensor(indices, device=DEVICE)
        return self._gather(indices_t)

    def update_priority_critic(self, indices: np.ndarray, priority: np.ndarray) -> None:
        self.max_priority = max(self.max_priority, float(priority.max()))
        self.critic_tree.set(indices, priority)

        top_value = self.critic_tree.total()
        self.actor_tree.set(indices, top_value / (priority + 1e-6))
