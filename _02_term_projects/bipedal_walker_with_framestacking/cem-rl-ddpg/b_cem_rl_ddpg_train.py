# CEM-RL (Pourchot & Sigaud, 2019) on top of DDPG.
#
# A diagonal Cross-Entropy-Method distribution N(mean, diag(var)) over the actor's flattened
# parameters samples a population of actors each generation. Half of them receive gradient steps
# from a shared DDPG critic before being evaluated; ALL of them are evaluated by REAL environment
# rollouts (fitness = episodic return) and their transitions fill a shared replay buffer. The CEM
# distribution is then refit to the top-half elites.
import os
import time
from datetime import datetime
from shutil import copyfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from a_ddpg_models import MODEL_DIR, Actor, QCritic, ReplayBuffer, DEVICE

import wandb


def make_env(env_name: str, stack_size: int, render_mode: str = None) -> gym.Env:
    env = gym.make(env_name, render_mode=render_mode)
    if stack_size and stack_size > 1:
        env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)
    return env


class CEM_RL_DDPG:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]
        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if use_wandb:
            self.wandb = wandb.init(project="cem_rl_ddpg_{0}".format(self.env_name), name=self.current_time, config=config)
        else:
            self.wandb = None

        self.max_generations = config["max_generations"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.soft_update_tau = config["soft_update_tau"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.learning_starts = config["learning_starts"]
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

        # actor used as the gradient container; eval_actor used purely for fitness rollouts
        self.actor = self._new_actor()
        self.target_actor = self._new_actor()
        self.eval_actor = self._new_actor()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        # shared critic (persists & accumulates learning across generations)
        self.q_critic = QCritic(self.n_features, self.n_actions, obs_ndim=self.obs_ndim)
        self.target_q_critic = QCritic(self.n_features, self.n_actions, obs_ndim=self.obs_ndim)
        self.target_q_critic.load_state_dict(self.q_critic.state_dict())
        self.q_critic_optimizer = optim.Adam(self.q_critic.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(
            capacity=self.replay_buffer_size, observation_shape=obs_shape, n_actions=self.n_actions
        )

        # CEM distribution over the flattened actor parameters
        self.mean = parameters_to_vector(self.actor.parameters()).detach().clone()
        self.n_params = self.mean.numel()
        self.var = torch.full_like(self.mean, self.sigma_init ** 2)

        self.time_steps = 0
        self.training_time_steps = 0
        self.total_train_start_time = None

    def _new_actor(self) -> Actor:
        return Actor(n_features=self.n_features, n_actions=self.n_actions, obs_ndim=self.obs_ndim)

    # ----------------------------------------------------------------- CEM sampling / update
    def sample_population(self) -> list[torch.Tensor]:
        std = (self.var + self.cem_noise).sqrt()
        return [self.mean + std * torch.randn(self.n_params, device=DEVICE) for _ in range(self.pop_size)]

    def cem_update(self, param_vectors: list[torch.Tensor], fitnesses: list[float]) -> None:
        order = list(np.argsort(fitnesses)[::-1])[: self.n_elite]
        # rank-based (CMA-ES style) weights for the elites
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
        """Load theta into the gradient actor, apply n_grad_steps DDPG updates, return improved params."""
        vector_to_parameters(theta_vec, self.actor.parameters())
        self.target_actor.load_state_dict(self.actor.state_dict())

        for _ in range(self.n_grad_steps):
            observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                next_mu_v = self.target_actor(next_observations)
                next_q_values = self.target_q_critic(next_observations, next_mu_v).squeeze(dim=-1)
                next_q_values[dones] = 0.0
                target_values = rewards.squeeze(dim=-1) + self.gamma * next_q_values

            q_values = self.q_critic(observations, actions).squeeze(dim=-1)
            critic_loss = F.mse_loss(q_values, target_values)
            self.q_critic_optimizer.zero_grad()
            critic_loss.backward()
            self.q_critic_optimizer.step()

            actor_loss = -1.0 * self.q_critic(observations, self.actor(observations)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_synchronize_models(self.actor, self.target_actor, self.soft_update_tau)
            self.soft_synchronize_models(self.q_critic, self.target_q_critic, self.soft_update_tau)
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

            # gradient half (only once the buffer has warmed up)
            do_grad = self.time_steps > self.learning_starts and self.replay_buffer.size() > self.batch_size
            if do_grad:
                for i in range(n_grad):
                    param_vectors[i] = self.gradient_steps(param_vectors[i])

            # evaluate ALL members via real env rollouts; fill the shared buffer
            fitnesses, steps_this_gen = [], 0
            for vec in param_vectors:
                fitness, steps = self.evaluate_params(vec, store=True)
                fitnesses.append(fitness)
                steps_this_gen += steps
            self.time_steps += steps_this_gen

            # refit CEM distribution to elites
            self.cem_update(param_vectors, fitnesses)

            # decay the CEM exploration-noise floor
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
                    "CEM noise: {:.2e},".format(self.cem_noise),
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
            "[TRAIN] Replay buffer": self.replay_buffer.size(),
            "generation": generation,
            "training steps": self.training_time_steps,
        })

    def model_save(self, validation_episode_reward_avg: float) -> None:
        vector_to_parameters(self.mean, self.eval_actor.parameters())
        filename = "cem_rl_ddpg_{0}_{1:4.1f}_{2}.pth".format(self.env_name, validation_episode_reward_avg, self.current_time)
        torch.save(self.eval_actor.state_dict(), os.path.join(MODEL_DIR, filename))
        copyfile(src=os.path.join(MODEL_DIR, filename), dst=os.path.join(MODEL_DIR, "cem_rl_ddpg_{0}_latest.pth".format(self.env_name)))

    def model_save_periodic(self, generation: int) -> None:
        vector_to_parameters(self.mean, self.eval_actor.parameters())
        filename = "cem_rl_ddpg_{0}_gen{1:06d}_{2}.pth".format(self.env_name, generation, self.current_time)
        torch.save(self.eval_actor.state_dict(), os.path.join(MODEL_DIR, filename))

    def validate(self) -> float:
        # validate the current CEM mean actor (deterministic)
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
        "max_generations": 5_000,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "soft_update_tau": 0.995,
        "replay_buffer_size": 1_000_000,
        "learning_starts": 10_000,
        "n_grad_steps": 200,             # gradient steps applied to each gradient-half actor
        # CEM
        "pop_size": 10,
        "n_elite": 5,                    # top half
        "eval_episodes": 1,
        "sigma_init": 1e-3,
        "noise_init": 0.05,
        "noise_end": 0.005,
        "noise_decay": 0.999,
        # logging / validation
        "print_generation_interval": 1,
        "validation_generation_interval": 10,
        "validation_num_episodes": 3,
        "episode_reward_avg_solved": 300,
        "save_generation_interval": 100,
    }

    env = make_env(ENV_NAME, config["stack_size"])
    test_env = make_env(ENV_NAME, config["stack_size"])

    use_wandb = True
    agent = CEM_RL_DDPG(env=env, test_env=test_env, config=config, use_wandb=use_wandb)
    agent.train_loop()


if __name__ == "__main__":
    main()
