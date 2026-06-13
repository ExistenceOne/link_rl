# CEM-RL (Pourchot & Sigaud, 2019) on top of SAC.
#
# A diagonal Cross-Entropy-Method distribution N(mean, diag(var)) over the GaussianPolicy's flattened
# parameters (both the mean and log_std heads) samples a population of actors each generation. Half of
# them receive SAC gradient steps (twin soft-Q critic + entropy term + automatic alpha) from a shared
# critic before being evaluated; ALL of them are evaluated by REAL environment rollouts using the
# deterministic mean action (fitness = episodic return) and their transitions fill a shared replay
# buffer. The CEM distribution is then refit to the top-half elites.
import os
import time
from datetime import datetime
from shutil import copyfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from a_sac_models import MODEL_DIR, GaussianPolicy, SoftQNetwork, ReplayBuffer, DEVICE

import wandb


def make_env(env_name: str, stack_size: int, render_mode: str = None) -> gym.Env:
    env = gym.make(env_name, render_mode=render_mode)
    if stack_size and stack_size > 1:
        env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)
    return env


class CEM_RL_SAC:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]
        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if use_wandb:
            self.wandb = wandb.init(project="cem_rl_sac_{0}".format(self.env_name), name=self.current_time, config=config)
        else:
            self.wandb = None

        self.max_generations = config["max_generations"]
        self.batch_size = config["batch_size"]
        self.policy_lr = config["policy_lr"]
        self.q_lr = config["q_lr"]
        self.alpha_lr = config["alpha_lr"]
        self.gamma = config["gamma"]
        self.soft_update_tau = config["soft_update_tau"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.learning_starts = config["learning_starts"]
        self.automatic_entropy_tuning = config["automatic_entropy_tuning"]
        self.n_grad_steps = config["n_grad_steps"]

        # CEM hyperparameters
        self.pop_size = config["pop_size"]
        self.n_elite = config["n_elite"]
        self.eval_episodes = config["eval_episodes"]
        self.sigma_init = config["sigma_init"]
        self.cem_noise = config["noise_init"]
        self.noise_init = config["noise_init"]
        self.noise_end = config["noise_end"]
        self.noise_decay = config["noise_decay"]

        self.print_generation_interval = config["print_generation_interval"]
        self.validation_generation_interval = config["validation_generation_interval"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]
        self.save_generation_interval = config["save_generation_interval"]

        obs_shape = env.observation_space.shape
        self.n_features = int(np.prod(obs_shape))
        self.obs_ndim = len(obs_shape)
        self.n_actions = env.action_space.shape[0]
        self.action_space = env.action_space

        self.actor = self._new_policy()
        self.eval_actor = self._new_policy()
        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=self.policy_lr)

        self.q_network = SoftQNetwork(self.n_features, self.n_actions, obs_ndim=self.obs_ndim)
        self.target_q_network = SoftQNetwork(self.n_features, self.n_actions, obs_ndim=self.obs_ndim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.q_network_optimizer = optim.Adam(self.q_network.parameters(), lr=self.q_lr)

        self.replay_buffer = ReplayBuffer(
            capacity=self.replay_buffer_size, observation_shape=obs_shape, n_actions=self.n_actions
        )

        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(DEVICE)).item()
            self.log_alpha = torch.tensor(0.2, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = 0.005
        self.max_alpha = 5.0

        # CEM distribution over the flattened policy parameters
        self.mean = parameters_to_vector(self.actor.parameters()).detach().clone()
        self.n_params = self.mean.numel()
        self.var = torch.full_like(self.mean, self.sigma_init ** 2)

        self.time_steps = 0
        self.training_time_steps = 0
        self.total_train_start_time = None

    def _new_policy(self) -> GaussianPolicy:
        return GaussianPolicy(
            n_features=self.n_features, n_actions=self.n_actions,
            action_space=self.action_space, obs_ndim=self.obs_ndim,
        )

    # ----------------------------------------------------------------- CEM sampling / update
    def sample_population(self) -> list[torch.Tensor]:
        std = (self.var + self.cem_noise).sqrt()
        return [self.mean + std * torch.randn(self.n_params, device=DEVICE) for _ in range(self.pop_size)]

    def cem_update(self, param_vectors: list[torch.Tensor], fitnesses: list[float]) -> None:
        order = list(np.argsort(fitnesses)[::-1])[: self.n_elite]
        ranks = torch.arange(1, self.n_elite + 1, device=DEVICE, dtype=torch.float32)
        weights = torch.log(torch.tensor(self.n_elite + 0.5, device=DEVICE)) - torch.log(ranks)
        weights = weights / weights.sum()

        old_mean = self.mean.clone()
        new_mean = torch.zeros_like(self.mean)
        new_var = torch.zeros_like(self.var)
        for w, idx in zip(weights, order):
            new_mean += w * param_vectors[idx]
        for w, idx in zip(weights, order):
            new_var += w * (param_vectors[idx] - old_mean) ** 2

        self.mean = new_mean
        self.var = new_var

    # ----------------------------------------------------------------- gradient phase
    def gradient_steps(self, theta_vec: torch.Tensor) -> torch.Tensor:
        vector_to_parameters(theta_vec, self.actor.parameters())

        for _ in range(self.n_grad_steps):
            observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(self.batch_size)

            # Q NETWORK UPDATE (target uses the current actor — SAC has no separate target actor)
            with torch.no_grad():
                next_state_action, next_state_log_pi, _, _ = self.actor.sample(next_observations)
                qf1_next_target, qf2_next_target = self.target_q_network(next_observations, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                min_qf_next_target[dones] = 0.0
                target_values = rewards + self.gamma * min_qf_next_target

            qf1, qf2 = self.q_network(observations, actions)
            qf_loss = F.mse_loss(qf1, target_values) + F.mse_loss(qf2, target_values)
            self.q_network_optimizer.zero_grad()
            qf_loss.backward()
            nn.utils.clip_grad_norm_(self.q_network.parameters(), 3.0)
            self.q_network_optimizer.step()

            # POLICY UPDATE
            sample_actions, log_pi, _, _ = self.actor.sample(observations, reparameterization_trick=True)
            qf1_pi, qf2_pi = self.q_network(observations, sample_actions)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            policy_loss = -1.0 * (min_qf_pi - self.alpha * log_pi).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 3.0)
            self.policy_optimizer.step()

            # ALPHA UPDATE
            if self.automatic_entropy_tuning:
                with torch.no_grad():
                    _, log_pi, _, _ = self.actor.sample(observations)
                alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                nn.utils.clip_grad_norm_([self.log_alpha], 3.0)
                self.alpha_optimizer.step()
                with torch.no_grad():
                    self.alpha = self.log_alpha.exp().item()
                    if self.alpha > self.max_alpha:
                        self.log_alpha.data.fill_(torch.log(torch.tensor(self.max_alpha)).item())

            self.soft_synchronize_models(self.q_network, self.target_q_network, self.soft_update_tau)
            self.training_time_steps += 1

        return parameters_to_vector(self.actor.parameters()).detach().clone()

    def soft_synchronize_models(self, source_model, target_model, tau):
        source_model_state = source_model.state_dict()
        target_model_state = target_model.state_dict()
        for k, v in source_model_state.items():
            target_model_state[k] = tau * target_model_state[k] + (1.0 - tau) * v
        target_model.load_state_dict(target_model_state)

    # ----------------------------------------------------------------- evaluation
    def evaluate_params(self, theta_vec: torch.Tensor, store: bool) -> tuple[float, int]:
        vector_to_parameters(theta_vec, self.eval_actor.parameters())
        rewards, total_steps = [], 0
        for _ in range(self.eval_episodes):
            observation, _ = self.env.reset()
            episode_reward, done = 0.0, False
            while not done:
                action = self.eval_actor.get_action(observation, exploration=False)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                if store:
                    self.replay_buffer.append(observation, action, next_observation, reward, terminated)
                episode_reward += reward
                total_steps += 1
                observation = next_observation
                done = terminated or truncated
            rewards.append(episode_reward)
        return float(np.mean(rewards)), total_steps

    # ----------------------------------------------------------------- main loop
    def train_loop(self) -> None:
        self.total_train_start_time = time.time()
        validation_episode_reward_avg = -200.0
        is_terminated = False
        n_grad = self.pop_size // 2

        for generation in range(1, self.max_generations + 1):
            param_vectors = self.sample_population()

            do_grad = self.time_steps > self.learning_starts and self.replay_buffer.size() > self.batch_size
            if do_grad:
                for i in range(n_grad):
                    param_vectors[i] = self.gradient_steps(param_vectors[i])

            fitnesses, steps_this_gen = [], 0
            for vec in param_vectors:
                fitness, steps = self.evaluate_params(vec, store=True)
                fitnesses.append(fitness)
                steps_this_gen += steps
            self.time_steps += steps_this_gen

            self.cem_update(param_vectors, fitnesses)
            self.cem_noise = max(self.noise_end, self.cem_noise * self.noise_decay)

            best_fitness = float(np.max(fitnesses))
            mean_fitness = float(np.mean(fitnesses))

            if generation % self.validation_generation_interval == 0:
                validation_episode_reward_avg = self.validate()
                if validation_episode_reward_avg > self.episode_reward_avg_solved:
                    self.model_save(validation_episode_reward_avg)
                    is_terminated = True

            if generation % self.save_generation_interval == 0:
                self.model_save_periodic(generation)

            if generation % self.print_generation_interval == 0:
                print(
                    "[Gen {:4,}, Time Steps {:7,}]".format(generation, self.time_steps),
                    "Pop Best: {:>8.2f}, Pop Mean: {:>8.2f},".format(best_fitness, mean_fitness),
                    "Alpha: {:>6.3f}, CEM noise: {:.2e},".format(self.alpha, self.cem_noise),
                    "Train Steps: {:6,}".format(self.training_time_steps),
                )

            if self.use_wandb:
                self.log_wandb(generation, validation_episode_reward_avg, best_fitness, mean_fitness)

            if is_terminated:
                print("Solved! Validation Episode Reward Avg: {0:.2f} > {1:.2f}".format(
                    validation_episode_reward_avg, self.episode_reward_avg_solved))
                break

        total_training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.total_train_start_time))
        print("Total Training End : {}".format(total_training_time))
        if self.use_wandb:
            self.wandb.finish()

    def log_wandb(self, generation, validation_episode_reward_avg, best_fitness, mean_fitness) -> None:
        self.wandb.log({
            "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(self.validation_num_episodes): validation_episode_reward_avg,
            "[EVO] population best fitness": best_fitness,
            "[EVO] population mean fitness": mean_fitness,
            "[CEM] noise": self.cem_noise,
            "[CEM] mean var": float(self.var.mean().item()),
            "[TRAIN] alpha": self.alpha,
            "[TRAIN] Replay buffer": self.replay_buffer.size(),
            "generation": generation,
            "training steps": self.training_time_steps,
        })

    def model_save(self, validation_episode_reward_avg: float) -> None:
        vector_to_parameters(self.mean, self.eval_actor.parameters())
        filename = "cem_rl_sac_{0}_{1:4.1f}_{2}.pth".format(self.env_name, validation_episode_reward_avg, self.current_time)
        torch.save(self.eval_actor.state_dict(), os.path.join(MODEL_DIR, filename))
        copyfile(src=os.path.join(MODEL_DIR, filename), dst=os.path.join(MODEL_DIR, "cem_rl_sac_{0}_latest.pth".format(self.env_name)))

    def model_save_periodic(self, generation: int) -> None:
        vector_to_parameters(self.mean, self.eval_actor.parameters())
        filename = "cem_rl_sac_{0}_gen{1:06d}_{2}.pth".format(self.env_name, generation, self.current_time)
        torch.save(self.eval_actor.state_dict(), os.path.join(MODEL_DIR, filename))

    def validate(self) -> float:
        vector_to_parameters(self.mean, self.eval_actor.parameters())
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)
        for i in range(self.validation_num_episodes):
            observation, _ = self.test_env.reset()
            episode_reward, done = 0.0, False
            while not done:
                action = self.eval_actor.get_action(observation, exploration=False)
                next_observation, reward, terminated, truncated, _ = self.test_env.step(action)
                episode_reward += reward
                observation = next_observation
                done = terminated or truncated
            episode_reward_lst[i] = episode_reward
        episode_reward_avg = float(np.average(episode_reward_lst))
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.total_train_start_time))
        print("[Validation Episode Reward: {0}] Average: {1:.3f}, Elapsed Time: {2}".format(
            episode_reward_lst, episode_reward_avg, elapsed))
        return episode_reward_avg


def main() -> None:
    print("TORCH VERSION:", torch.__version__, "| DEVICE:", DEVICE)
    ENV_NAME = "BipedalWalkerHardcore-v3"

    config = {
        "env_name": ENV_NAME,
        "stack_size": 1,                 # 1 disables frame stacking; >1 stacks that many frames
        "max_generations": 10_000,
        "batch_size": 256,
        "policy_lr": 7e-4,
        "q_lr": 7e-4,
        "alpha_lr": 7e-4,
        "gamma": 0.99,
        "soft_update_tau": 0.99,
        "replay_buffer_size": 1_000_000,
        "learning_starts": 1_000,
        "automatic_entropy_tuning": True,
        "n_grad_steps": 200,
        # CEM
        "pop_size": 10,
        "n_elite": 5,
        "eval_episodes": 1,
        "sigma_init": 1e-3,
        "noise_init": 1e-3,
        "noise_end": 1e-5,
        "noise_decay": 0.999,
        # logging / validation
        "print_generation_interval": 1,
        "validation_generation_interval": 10,
        "validation_num_episodes": 3,
        "episode_reward_avg_solved": 300,
        "save_generation_interval": 10,
    }

    env = make_env(ENV_NAME, config["stack_size"])
    test_env = make_env(ENV_NAME, config["stack_size"])

    use_wandb = True
    agent = CEM_RL_SAC(env=env, test_env=test_env, config=config, use_wandb=use_wandb)
    agent.train_loop()


if __name__ == "__main__":
    main()
