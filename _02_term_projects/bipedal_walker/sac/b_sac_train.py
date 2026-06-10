# https://gymnasium.farama.org/environments/classic_control/cart_pole/
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
from gymnasium.wrappers import NormalizeReward

from a_sac_models import MODEL_DIR, GaussianPolicy, SoftQNetwork, ReplayBuffer, Transition, DEVICE

import wandb


class SAC:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]

        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if use_wandb:
            self.wandb = wandb.init(project="sac_{0}".format(self.env_name), name=self.current_time, config=config)
        else:
            self.wandb = None

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.print_episode_interval = config["print_episode_interval"]
        self.validation_time_steps_interval = config["validation_time_steps_interval"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]
        self.steps_between_train = config["steps_between_train"]
        self.soft_update_tau = config["soft_update_tau"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.learning_starts = config["learning_starts"]
        self.automatic_entropy_tuning = config["automatic_entropy_tuning"]

        # ERE (Emphasizing Recent Experience, Wang & Ross 2019)
        self.use_ere = config["use_ere"]
        self.ere_eta = config["ere_eta"]            # recency emphasis (closer to 1.0 == more uniform)
        self.ere_min_size = config["ere_min_size"]  # c_min: floor on the recent-window size

        # ES (Evolution Strategies with critic-surrogate fitness)
        self.use_es = config["use_es"]
        self.es_num_perturbations = config["es_num_perturbations"]  # K, must be even (antithetic pairs)
        self.es_sigma = config["es_sigma"]          # parameter noise std
        self.es_lr = config["es_lr"]                # ES step size

        n_features = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        self.policy = GaussianPolicy(n_features=n_features, n_actions=n_actions, action_space=env.action_space)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        self.q_network = SoftQNetwork(n_features=n_features, n_actions=n_actions)
        self.target_q_network = SoftQNetwork(n_features=n_features, n_actions=n_actions)

        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.q_network_optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)

        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(DEVICE)).item()
            print("TARGET ENTROPY: {0}".format(self.target_entropy))
            self.log_alpha = torch.tensor(0.2, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = 0.2

        self.time_steps = 0
        self.training_time_steps = 0

        self.max_alpha = 5.0

        self.total_train_start_time = None

    def train_loop(self) -> None:
        self.total_train_start_time = time.time()

        validation_episode_reward_avg = -1500
        policy_loss = q_1_td_loss = q_2_td_loss = alpha_loss = mu = entropy = es_fitness = 0.0

        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            episode_reward = 0
            episode_steps = 0

            observation, _ = self.env.reset()

            done = False

            while not done:
                self.time_steps += 1
                episode_steps += 1

                if self.time_steps < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action = self.policy.get_action(observation)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.replay_buffer.append(transition)

                observation = next_observation
                done = terminated or truncated

                # Uniform-replay path: one update every `steps_between_train` env steps.
                # With ERE the updates are deferred to an end-of-episode burst (see below).
                if not self.use_ere and self.time_steps % self.steps_between_train == 0 and self.time_steps > self.batch_size:
                    policy_loss, q_1_td_loss, q_2_td_loss, alpha_loss, mu, entropy, es_fitness = self.train()

                if self.time_steps % self.validation_time_steps_interval == 0:
                    validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

                    if validation_episode_reward_avg > self.episode_reward_avg_solved:
                        print("Solved in {0:,} time steps ({1:,} training steps)!".format(self.time_steps, self.training_time_steps))
                        self.model_save(validation_episode_reward_avg)
                        is_terminated = True

                    if self.use_wandb:
                        self.log_wandb(
                            validation_episode_reward_avg,
                            episode_reward,
                            policy_loss,
                            q_1_td_loss, q_2_td_loss,
                            alpha_loss,
                            mu,
                            entropy,
                            es_fitness,
                            n_episode,
                        )

            # ERE: after each episode of length T, run a burst of T // steps_between_train updates,
            # each sampling from a progressively shrinking most-recent window of the replay buffer.
            if self.use_ere and self.time_steps > self.learning_starts and self.replay_buffer.size() > self.batch_size:
                num_updates = max(1, episode_steps // self.steps_between_train)
                policy_loss, q_1_td_loss, q_2_td_loss, alpha_loss, mu, entropy, es_fitness = self.train_ere(num_updates)

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Epi. {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "Epi. Reward: {:>9.3f},".format(episode_reward),
                    "Policy L.: {:>7.3f},".format(policy_loss),
                    "Critic L.: {:>7.3f}, {:>7.3f}".format(q_1_td_loss, q_2_td_loss),
                    "Alpha L.: {:>7.3f},".format(alpha_loss),
                    "Alpha: {:>7.3f},".format(self.alpha),
                    "Entropy: {:>7.3f},".format(entropy),
                    "Train Steps: {:5,}".format(self.training_time_steps),
                )

            if is_terminated:
                if self.wandb:
                    for _ in range(5):
                        self.log_wandb(
                            validation_episode_reward_avg,
                            episode_reward,
                            policy_loss,
                            q_1_td_loss, q_2_td_loss,
                            alpha_loss,
                            mu,
                            entropy,
                            es_fitness,
                            n_episode,
                        )
                break

        total_training_time = time.time() - self.total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))
        if self.use_wandb:
            self.wandb.finish()

    def log_wandb(
        self,
        validation_episode_reward_avg: float,
        episode_reward: float,
        policy_loss: float,
        q_1_td_loss: float, q_2_td_loss: float,
        alpha_loss: float,
        mu: float,
        entropy: float,
        es_fitness: float,
        n_episode: float,
    ) -> None:
        self.wandb.log(
            {
                "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(
                    self.validation_num_episodes
                ): validation_episode_reward_avg,
                "[TRAIN] episode reward": episode_reward,
                "[TRAIN] policy loss": policy_loss,
                "[TRAIN] critic 1 loss": q_1_td_loss,
                "[TRAIN] critic 2 loss": q_2_td_loss,
                "[TRAIN] alpha loss": alpha_loss,
                "[TRAIN] alpha": self.alpha,
                "[TRAIN] mu": mu,
                "[TRAIN] entropy": entropy,
                "[TRAIN] ES fitness": es_fitness,
                "[TRAIN] Replay buffer": self.replay_buffer.size(),
                "training episode": n_episode,
                "training steps": self.training_time_steps,
            }
        )

    def train_ere(self, num_updates: int):
        # N: current number of transitions in the replay buffer.
        buffer_size = self.replay_buffer.size()

        policy_loss = q_1_td_loss = q_2_td_loss = alpha_loss = mu = entropy = es_fitness = 0.0

        for k in range(num_updates):
            # c_k = N * eta^(k * 1000 / K), floored at c_min. The k=0 update sees the whole
            # buffer; later updates focus on increasingly recent experience.
            if self.ere_eta < 1.0:
                c_k = int(buffer_size * (self.ere_eta ** (k * 1000.0 / num_updates)))
                c_k = max(c_k, self.ere_min_size)
            else:
                c_k = buffer_size

            policy_loss, q_1_td_loss, q_2_td_loss, alpha_loss, mu, entropy, es_fitness = self.train(most_recent=c_k)

        return policy_loss, q_1_td_loss, q_2_td_loss, alpha_loss, mu, entropy, es_fitness

    def train(self, most_recent: int = None):
        self.training_time_steps += 1

        observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(
            self.batch_size, most_recent=most_recent
        )

        ####################
        # Q NETWORK UPDATE #
        ####################
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_observations)
            qf1_next_target, qf2_next_target = self.target_q_network(next_observations, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            min_qf_next_target[dones] = 0.0
            target_values = rewards + self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.q_network(observations, actions)
        qf1_loss = F.mse_loss(qf1, target_values)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, target_values)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.q_network_optimizer.zero_grad()
        qf_loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 3.0)
        self.q_network_optimizer.step()

        #################
        # Policy UPDATE #
        #################
        sample_actions, log_pi, mu, entropy = self.policy.sample(observations, reparameterization_trick=True)

        qf1_pi, qf2_pi = self.q_network(observations, sample_actions)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = -1.0 * (min_qf_pi - self.alpha * log_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        #print(min_qf_pi.max(), self.alpha, log_pi.max(), (min_qf_pi - self.alpha * log_pi).mean(), "!!!!!!!!!!!!!!!!!!!!!!!!")
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 3.0)
        self.policy_optimizer.step()

        #################
        # Alpha UPDATE #
        #################
        if self.automatic_entropy_tuning:
            with torch.no_grad():
                _, log_pi, _, _ = self.policy.sample(observations)

            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()

            # print(self.target_entropy, (log_pi + self.target_entropy).detach().mean(), self.log_alpha.exp(), alpha_loss, "!!!!!!!!!!!!!!")
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            nn.utils.clip_grad_norm_([self.log_alpha], 3.0)
            self.alpha_optimizer.step()

            # log_alpha를 기반으로 알파 값 계산 (상한선을 설정)
            with torch.no_grad():
                self.alpha = self.log_alpha.exp().item()

                # 클램핑 기법을 사용해 알파 값이 상한선을 넘지 않도록 제한
                if self.alpha > self.max_alpha:
                    self.log_alpha.data.fill_(torch.log(torch.tensor(self.max_alpha)).item())
        else:
            alpha_loss = torch.tensor(0.).to(DEVICE)

        # sync, TAU: 0.995
        self.soft_synchronize_models(
            source_model=self.q_network, target_model=self.target_q_network, tau=self.soft_update_tau
        )

        # ES update: zero-order gradient via antithetic critic-surrogate fitness.
        # Runs after the gradient step so Q-network weights are current.
        es_fitness = 0.0
        if self.use_es:
            es_fitness = self._es_update(observations)

        return policy_loss.item(), qf1_loss.item(), qf2_loss.item(), alpha_loss.item(), mu.mean().item(), entropy.item(), es_fitness

    def _es_update(self, observations: torch.Tensor) -> float:
        """
        Antithetic ES update using min(Q1, Q2) as surrogate fitness.
        K/2 noise vectors are mirrored (+ε, -ε) to reduce variance.
        No environment steps are needed — the critic acts as the fitness function.
        """
        param_list = list(self.policy.parameters())
        saved = [p.data.clone() for p in param_list]
        half_k = max(1, self.es_num_perturbations // 2)

        noises = [[torch.randn_like(p) for p in param_list] for _ in range(half_k)]

        pos_fitnesses, neg_fitnesses = [], []

        for eps in noises:
            # Positive perturbation: θ + σε
            for p, e, s in zip(param_list, eps, saved):
                p.data.copy_(s + self.es_sigma * e)
            with torch.no_grad():
                actions, _, _, _ = self.policy.sample(observations)
                q1, q2 = self.q_network(observations, actions)
                pos_fitnesses.append(torch.min(q1, q2).mean().item())

            # Negative perturbation: θ - σε
            for p, e, s in zip(param_list, eps, saved):
                p.data.copy_(s - self.es_sigma * e)
            with torch.no_grad():
                actions, _, _, _ = self.policy.sample(observations)
                q1, q2 = self.q_network(observations, actions)
                neg_fitnesses.append(torch.min(q1, q2).mean().item())

        # Restore original parameters before applying the update
        for p, s in zip(param_list, saved):
            p.data.copy_(s)

        # Fitness shaping: normalize by std of all evaluations for scale stability
        all_f = np.array(pos_fitnesses + neg_fitnesses)
        f_std = all_f.std() + 1e-8
        diffs = (np.array(pos_fitnesses) - np.array(neg_fitnesses)) / f_std

        # ES gradient estimate (antithetic): Δθ = (es_lr / (K * σ)) * Σ_i diff_i * ε_i
        scale = self.es_lr / (self.es_num_perturbations * self.es_sigma)
        for i, eps in enumerate(noises):
            for p, e in zip(param_list, eps):
                p.data.add_(scale * float(diffs[i]) * e)

        return float(all_f.mean())

    def soft_synchronize_models(self, source_model, target_model, tau):
        source_model_state = source_model.state_dict()
        target_model_state = target_model.state_dict()
        for k, v in source_model_state.items():
            target_model_state[k] = tau * target_model_state[k] + (1.0 - tau) * v
        target_model.load_state_dict(target_model_state)

    def model_save(self, validation_episode_reward_avg: float) -> None:
        filename = "sac_{0}_{1:4.1f}_{2}.pth".format(self.env_name, validation_episode_reward_avg, self.current_time)
        torch.save(self.policy.state_dict(), os.path.join(MODEL_DIR, filename))

        copyfile(src=os.path.join(MODEL_DIR, filename), dst=os.path.join(MODEL_DIR, "sac_{0}_latest.pth".format(self.env_name)))

    def validate(self) -> tuple[np.ndarray, float]:
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        for i in range(self.validation_num_episodes):
            episode_reward = 0

            observation, _ = self.test_env.reset()

            done = False

            while not done:
                action = self.policy.get_action(observation, exploration=False)

                next_observation, reward, terminated, truncated, _ = self.test_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

        episode_reward_avg = np.average(episode_reward_lst)

        total_training_time = time.time() - self.total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))

        print(
            "[Validation Episode Reward: {0}] Average: {1:.3f}, Elapsed Time: {2}".format(
                episode_reward_lst, episode_reward_avg, total_training_time
            )
        )
        return episode_reward_lst, episode_reward_avg


def main() -> None:
    print("TORCH VERSION:", torch.__version__)
    ENV_NAME = "BipedalWalkerHardcore-v3"

    # env
    env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)

    config = {
        "env_name": ENV_NAME,
        "max_num_episodes": 200_000,
        "batch_size": 256,
        "steps_between_train": 1,
        "replay_buffer_size": 1_000_000,
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "soft_update_tau": 0.99,
        "print_episode_interval": 20,
        "validation_time_steps_interval": 25_000,
        "validation_num_episodes": 3,
        "episode_reward_avg_solved": 300,
        "learning_starts": 10_000,
        "automatic_entropy_tuning": True,
        "use_ere": False,
        "ere_eta": 0.996,
        "ere_min_size": 5_000,
        "use_es": True,
        "es_num_perturbations": 20,  # K (antithetic pairs: K/2)
        "es_sigma": 0.01,            # parameter noise std
        "es_lr": 1e-3,               # ES step size
    }

    use_wandb = True
    sac = SAC(env=env, test_env=test_env, config=config, use_wandb=use_wandb)
    sac.train_loop()


if __name__ == "__main__":
    main()
