# Evolutionary Reinforcement Learning (ERL, Khadka & Tumer 2018) on top of SAC.
#
# A GA-evolved population of GaussianPolicy actors plus one gradient-based SAC learner share a
# single replay buffer. Population members are evaluated by REAL environment rollouts (fitness =
# episodic return); their transitions fill the buffer, the SAC learner trains from the buffer, and
# the SAC actor is periodically injected back into the population.
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

from a_sac_models import MODEL_DIR, GaussianPolicy, SoftQNetwork, ReplayBuffer, DEVICE

import wandb


def make_env(env_name: str, stack_size: int, render_mode: str = None) -> gym.Env:
    env = gym.make(env_name, render_mode=render_mode)
    if stack_size and stack_size > 1:
        env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)
    return env


class ERL_SAC:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]
        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if use_wandb:
            self.wandb = wandb.init(project="erl_sac_{0}".format(self.env_name), name=self.current_time, config=config)
        else:
            self.wandb = None

        # RL hyperparameters
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
        self.grad_steps_ratio = config["grad_steps_ratio"]

        # ERL (evolution) hyperparameters
        self.pop_size = config["pop_size"]
        self.n_elite = config["n_elite"]
        self.eval_episodes = config["eval_episodes"]
        self.mut_prob = config["mut_prob"]
        self.mut_strength = config["mut_strength"]
        self.tournament_k = config["tournament_k"]
        self.sync_period = config["sync_period"]

        # bookkeeping / logging
        self.print_generation_interval = config["print_generation_interval"]
        self.validation_generation_interval = config["validation_generation_interval"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_save = config["episode_reward_avg_save"]

        obs_shape = env.observation_space.shape
        self.n_features = int(np.prod(obs_shape))
        self.obs_ndim = len(obs_shape)
        self.n_actions = env.action_space.shape[0]
        self.action_space = env.action_space

        # SAC learner
        self.policy = self._new_policy()
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)

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
            self.alpha = 0.2
        self.max_alpha = 5.0

        # Population of additional actors evolved by the GA
        self.population = [self._new_policy() for _ in range(self.pop_size)]

        self.time_steps = 0
        self.training_time_steps = 0
        self.total_train_start_time = None

    def _new_policy(self) -> GaussianPolicy:
        return GaussianPolicy(
            n_features=self.n_features, n_actions=self.n_actions,
            action_space=self.action_space, obs_ndim=self.obs_ndim,
        )

    # ----------------------------------------------------------------- rollouts
    def rollout(self, actor: GaussianPolicy, exploration: bool, store: bool) -> tuple[float, int]:
        observation, _ = self.env.reset()
        episode_reward, steps = 0.0, 0
        done = False
        while not done:
            action = actor.get_action(observation, exploration=exploration)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            if store:
                self.replay_buffer.append(observation, action, next_observation, reward, terminated)
            episode_reward += reward
            steps += 1
            observation = next_observation
            done = terminated or truncated
        return episode_reward, steps

    def evaluate_actor(self, actor: GaussianPolicy, exploration: bool, store: bool) -> tuple[float, int]:
        rewards, total_steps = [], 0
        for _ in range(self.eval_episodes):
            r, s = self.rollout(actor, exploration=exploration, store=store)
            rewards.append(r)
            total_steps += s
        return float(np.mean(rewards)), total_steps

    # ----------------------------------------------------------------- SAC update
    def sac_train_step(self):
        self.training_time_steps += 1
        observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(self.batch_size)

        # Q NETWORK UPDATE
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_observations)
            qf1_next_target, qf2_next_target = self.target_q_network(next_observations, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            min_qf_next_target[dones] = 0.0
            target_values = rewards + self.gamma * min_qf_next_target

        qf1, qf2 = self.q_network(observations, actions)
        qf1_loss = F.mse_loss(qf1, target_values)
        qf2_loss = F.mse_loss(qf2, target_values)
        qf_loss = qf1_loss + qf2_loss

        self.q_network_optimizer.zero_grad()
        qf_loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 3.0)
        self.q_network_optimizer.step()

        # POLICY UPDATE
        sample_actions, log_pi, mu, entropy = self.policy.sample(observations, reparameterization_trick=True)
        qf1_pi, qf2_pi = self.q_network(observations, sample_actions)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = -1.0 * (min_qf_pi - self.alpha * log_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 3.0)
        self.policy_optimizer.step()

        # ALPHA UPDATE
        if self.automatic_entropy_tuning:
            with torch.no_grad():
                _, log_pi, _, _ = self.policy.sample(observations)
            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            nn.utils.clip_grad_norm_([self.log_alpha], 3.0)
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.alpha = self.log_alpha.exp().item()
                if self.alpha > self.max_alpha:
                    self.log_alpha.data.fill_(torch.log(torch.tensor(self.max_alpha)).item())
        else:
            alpha_loss = torch.tensor(0.).to(DEVICE)

        self.soft_synchronize_models(self.q_network, self.target_q_network, self.soft_update_tau)

        return policy_loss.item(), qf1_loss.item(), qf2_loss.item(), alpha_loss.item(), entropy.item()

    def soft_synchronize_models(self, source_model, target_model, tau):
        source_model_state = source_model.state_dict()
        target_model_state = target_model.state_dict()
        for k, v in source_model_state.items():
            target_model_state[k] = tau * target_model_state[k] + (1.0 - tau) * v
        target_model.load_state_dict(target_model_state)

    # ----------------------------------------------------------------- evolution
    def _tournament_select(self, fitnesses) -> int:
        idxs = np.random.randint(0, len(fitnesses), size=self.tournament_k)
        return int(idxs[int(np.argmax([fitnesses[i] for i in idxs]))])

    def _crossover(self, parent_a: GaussianPolicy, parent_b: GaussianPolicy) -> dict:
        sa, sb = parent_a.state_dict(), parent_b.state_dict()
        child = {}
        for k in sa:
            mask = torch.rand_like(sa[k]) < 0.5
            child[k] = torch.where(mask, sa[k], sb[k]).clone()
        return child

    def _mutate(self, state: dict) -> None:
        for k in state:
            mask = (torch.rand_like(state[k]) < self.mut_prob).float()
            state[k] = state[k] + mask * torch.randn_like(state[k]) * self.mut_strength

    def evolve(self, fitnesses) -> None:
        order = list(np.argsort(fitnesses)[::-1])  # descending by fitness
        new_states = []
        # elitism: carry the top n_elite unchanged
        for e in order[: self.n_elite]:
            new_states.append({k: v.clone() for k, v in self.population[e].state_dict().items()})
        # offspring: tournament selection -> crossover -> mutation
        while len(new_states) < self.pop_size:
            pa = self.population[self._tournament_select(fitnesses)]
            pb = self.population[self._tournament_select(fitnesses)]
            child = self._crossover(pa, pb)
            self._mutate(child)
            new_states.append(child)
        for actor, st in zip(self.population, new_states):
            actor.load_state_dict(st)

    def inject_rl_actor(self) -> None:
        # Replace a non-elite slot (the last one) with the gradient-trained SAC actor.
        self.population[-1].load_state_dict(self.policy.state_dict())

    # ----------------------------------------------------------------- main loop
    def train_loop(self) -> None:
        self.total_train_start_time = time.time()
        validation_episode_reward_avg = -1000.0
        policy_loss = q1_loss = q2_loss = alpha_loss = entropy = 0.0

        for generation in range(1, self.max_generations + 1):
            # 1) evaluate the population via real env rollouts; fill the shared buffer
            fitnesses, steps_this_gen = [], 0
            for actor in self.population:
                fitness, steps = self.evaluate_actor(actor, exploration=False, store=True)
                fitnesses.append(fitness)
                steps_this_gen += steps
            self.time_steps += steps_this_gen

            # 2) RL actor rollout (stochastic exploration)
            rl_return, rl_steps = self.rollout(self.policy, exploration=True, store=True)
            self.time_steps += rl_steps
            steps_this_gen += rl_steps

            # 3) SAC gradient updates from the shared buffer
            if self.time_steps > self.learning_starts and self.replay_buffer.size() > self.batch_size:
                num_updates = max(1, int(steps_this_gen * self.grad_steps_ratio))
                for _ in range(num_updates):
                    policy_loss, q1_loss, q2_loss, alpha_loss, entropy = self.sac_train_step()

            # 4) evolve the population
            self.evolve(fitnesses)

            # 5) periodically inject the SAC actor into the population
            if generation % self.sync_period == 0:
                self.inject_rl_actor()

            best_fitness = float(np.max(fitnesses))
            mean_fitness = float(np.mean(fitnesses))

            # validation + checkpoint
            if generation % self.validation_generation_interval == 0:
                validation_episode_reward_avg = self.validate()
                if validation_episode_reward_avg > self.episode_reward_avg_save:
                    self.model_save(validation_episode_reward_avg)

            if generation % self.print_generation_interval == 0:
                print(
                    "[Gen {:4,}, Time Steps {:7,}]".format(generation, self.time_steps),
                    "Pop Best: {:>8.2f}, Pop Mean: {:>8.2f},".format(best_fitness, mean_fitness),
                    "RL Return: {:>8.2f},".format(rl_return),
                    "Policy L.: {:>7.3f}, Critic L.: {:>7.3f}/{:>7.3f},".format(policy_loss, q1_loss, q2_loss),
                    "Alpha: {:>6.3f}, Train Steps: {:6,}".format(self.alpha, self.training_time_steps),
                )

            if self.use_wandb:
                self.log_wandb(
                    generation, validation_episode_reward_avg, best_fitness, mean_fitness,
                    rl_return, policy_loss, q1_loss, q2_loss, alpha_loss, entropy,
                )

        total_training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.total_train_start_time))
        print("Total Training End : {}".format(total_training_time))
        if self.use_wandb:
            self.wandb.finish()

    def log_wandb(self, generation, validation_episode_reward_avg, best_fitness, mean_fitness,
                  rl_return, policy_loss, q1_loss, q2_loss, alpha_loss, entropy) -> None:
        self.wandb.log({
            "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(self.validation_num_episodes): validation_episode_reward_avg,
            "[EVO] population best fitness": best_fitness,
            "[EVO] population mean fitness": mean_fitness,
            "[TRAIN] rl actor return": rl_return,
            "[TRAIN] policy loss": policy_loss,
            "[TRAIN] critic 1 loss": q1_loss,
            "[TRAIN] critic 2 loss": q2_loss,
            "[TRAIN] alpha loss": alpha_loss,
            "[TRAIN] alpha": self.alpha,
            "[TRAIN] entropy": entropy,
            "[TRAIN] Replay buffer": self.replay_buffer.size(),
            "generation": generation,
            "training steps": self.training_time_steps,
        })

    def model_save(self, validation_episode_reward_avg: float) -> None:
        filename = "erl_sac_{0}_{1:4.1f}_{2}.pth".format(self.env_name, validation_episode_reward_avg, self.current_time)
        torch.save(self.policy.state_dict(), os.path.join(MODEL_DIR, filename))
        copyfile(src=os.path.join(MODEL_DIR, filename), dst=os.path.join(MODEL_DIR, "erl_sac_{0}_latest.pth".format(self.env_name)))

    def validate(self) -> float:
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)
        for i in range(self.validation_num_episodes):
            observation, _ = self.test_env.reset()
            episode_reward, done = 0.0, False
            while not done:
                action = self.policy.get_action(observation, exploration=False)
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
        "stack_size": 4,                 # 1 disables frame stacking; >1 stacks that many frames
        "max_generations": 5_000,
        "batch_size": 256,
        "policy_lr": 3e-4,
        "q_lr": 3e-4,
        "alpha_lr": 3e-4,
        "gamma": 0.99,
        "soft_update_tau": 0.995,
        "replay_buffer_size": 1_000_000,
        "learning_starts": 10_000,
        "automatic_entropy_tuning": True,
        "grad_steps_ratio": 1.0,         # SAC gradient steps per env step collected each generation
        # ERL / evolution
        "pop_size": 10,
        "n_elite": 2,
        "eval_episodes": 1,
        "mut_prob": 0.1,
        "mut_strength": 0.1,
        "tournament_k": 3,
        "sync_period": 1,
        # logging / validation
        "print_generation_interval": 1,
        "validation_generation_interval": 10,
        "validation_num_episodes": 3,
        "episode_reward_avg_save": 0,
    }

    env = make_env(ENV_NAME, config["stack_size"])
    test_env = make_env(ENV_NAME, config["stack_size"])

    use_wandb = True
    agent = ERL_SAC(env=env, test_env=test_env, config=config, use_wandb=use_wandb)
    agent.train_loop()


if __name__ == "__main__":
    main()
