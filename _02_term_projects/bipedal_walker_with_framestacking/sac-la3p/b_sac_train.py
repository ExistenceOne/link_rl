# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os
import time
from datetime import datetime
from shutil import copyfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from a_sac_models import MODEL_DIR, GaussianPolicy, SoftQNetwork, LA3PReplayBuffer, DEVICE

import wandb


def _huber(td_error: torch.Tensor, min_priority: float) -> torch.Tensor:
    """Huber loss of a non-negative TD-error magnitude (LA3P's critic-prioritized loss)."""
    return torch.where(td_error < min_priority, 0.5 * td_error.pow(2), min_priority * td_error).mean()


def _pal(td_loss: torch.Tensor, min_priority: float, alpha: float) -> torch.Tensor:
    """Prioritized Approximation Loss (LA3P's uniform-batch critic loss)."""
    abs_loss = td_loss.abs()
    return torch.where(
        abs_loss < min_priority,
        (min_priority ** alpha) * 0.5 * td_loss.pow(2),
        min_priority * abs_loss.pow(1.0 + alpha) / (1.0 + alpha),
    ).mean()


class SAC:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]

        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if use_wandb:
            self.wandb = wandb.init(project="sac_la3p_{0}".format(self.env_name), name=self.current_time, config=config)
        else:
            self.wandb = None

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.policy_lr = config["policy_lr"]
        self.q_lr = config["q_lr"]
        self.alpha_lr = config["alpha_lr"]
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

        self.la3p_alpha = config["la3p_alpha"]
        self.la3p_min_priority = config["la3p_min_priority"]
        self.la3p_prioritized_fraction = config["la3p_prioritized_fraction"]

        n_features = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        self.policy = GaussianPolicy(n_features=n_features, n_actions=n_actions, action_space=env.action_space)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)

        self.q_network = SoftQNetwork(n_features=n_features, n_actions=n_actions)
        self.target_q_network = SoftQNetwork(n_features=n_features, n_actions=n_actions)

        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.q_network_optimizer = optim.Adam(self.q_network.parameters(), lr=self.q_lr)

        self.replay_buffer = LA3PReplayBuffer(
            capacity=self.replay_buffer_size, n_features=n_features, n_actions=n_actions,
            alpha=self.la3p_alpha, min_priority=self.la3p_min_priority,
        )

        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(DEVICE)).item()
            print("TARGET ENTROPY: {0}".format(self.target_entropy))
            self.log_alpha = torch.tensor(-1.6, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = 0.005

        self.time_steps = 0
        self.training_time_steps = 0

        self.max_alpha = 5.0

        self.total_train_start_time = None

    def train_loop(self) -> None:
        self.total_train_start_time = time.time()

        validation_episode_reward_avg = -100
        policy_loss = q_1_td_loss = q_2_td_loss = alpha_loss = mu = entropy = 0.0

        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            episode_reward = 0

            observation, _ = self.env.reset()

            done = False

            while not done:
                self.time_steps += 1

                if self.time_steps < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action = self.policy.get_action(observation)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward

                self.replay_buffer.append(observation, action, next_observation, reward, terminated)

                observation = next_observation
                done = terminated or truncated

                if self.time_steps % self.steps_between_train == 0 and self.time_steps > self.batch_size:
                    policy_loss, q_1_td_loss, q_2_td_loss, alpha_loss, mu, entropy = self.train()

                if self.time_steps % self.validation_time_steps_interval == 0:
                    validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

                    self.model_save(validation_episode_reward_avg)
                    if validation_episode_reward_avg > self.episode_reward_avg_solved:
                        print("Solved in {0:,} time steps ({1:,} training steps)!".format(self.time_steps, self.training_time_steps))
                        is_terminated = True


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
                if self.use_wandb:
                    self.log_wandb(
                        validation_episode_reward_avg,
                        episode_reward,
                        policy_loss,
                        q_1_td_loss, q_2_td_loss,
                        alpha_loss,
                        mu,
                        entropy,
                        n_episode,
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
                "[TRAIN] Replay buffer": self.replay_buffer.size(),
                "[TRAIN] LA3P max priority": self.replay_buffer.max_priority,
                "training episode": n_episode,
                "training steps": self.training_time_steps,
            }
        )

    def _train_critic(self, batch, uniform: bool):
        observations, actions, next_observations, rewards, dones, indices = batch

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_observations)
            qf1_next_target, qf2_next_target = self.target_q_network(next_observations, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            min_qf_next_target[dones] = 0.0
            target_values = rewards + self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.q_network(observations, actions)
        td_loss_1 = qf1 - target_values
        td_loss_2 = qf2 - target_values
        td_error_1 = td_loss_1.abs()
        td_error_2 = td_loss_2.abs()

        if uniform:
            # Prioritized Approximation Loss, normalized so the gradient matches the scale of
            # the loss the critic-prioritized batch would have produced for these transitions.
            normalizer = torch.max(td_error_1, td_error_2).clamp(min=self.la3p_min_priority).pow(self.la3p_alpha).mean().detach()
            qf1_loss = _pal(td_loss_1, self.la3p_min_priority, self.la3p_alpha) / normalizer
            qf2_loss = _pal(td_loss_2, self.la3p_min_priority, self.la3p_alpha) / normalizer
        else:
            qf1_loss = _huber(td_error_1, self.la3p_min_priority)
            qf2_loss = _huber(td_error_2, self.la3p_min_priority)

        qf_loss = qf1_loss + qf2_loss

        self.q_network_optimizer.zero_grad()
        qf_loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 3.0)
        self.q_network_optimizer.step()

        with torch.no_grad():
            priority = torch.max(td_error_1, td_error_2).clamp(min=self.la3p_min_priority).pow(self.la3p_alpha).cpu().numpy().flatten()
        self.replay_buffer.update_priority_critic(indices, priority)

        return qf1_loss.item(), qf2_loss.item()

    def _train_actor(self, observations):
        sample_actions, log_pi, mu, entropy = self.policy.sample(observations, reparameterization_trick=True)

        qf1_pi, qf2_pi = self.q_network(observations, sample_actions)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = -1.0 * (min_qf_pi - self.alpha * log_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 3.0)
        self.policy_optimizer.step()

        #################
        # Alpha UPDATE #
        #################
        if self.automatic_entropy_tuning:
            with torch.no_grad():
                _, log_pi_for_alpha, _, _ = self.policy.sample(observations)

            alpha_loss = (-self.log_alpha.exp() * (log_pi_for_alpha + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            nn.utils.clip_grad_norm_([self.log_alpha], 3.0)
            self.alpha_optimizer.step()

            # log_alpha를 기반으로 알파 값 계산 (상한선을 설정)
            with torch.no_grad():
                self.alpha = self.log_alpha.exp().item()

                # 클램핑 기법을 사용해 알파 값이 상한선을 넘지 않도록 제한
                if self.alpha > self.max_alpha:
                    self.log_alpha.data = torch.log(torch.tensor(self.max_alpha, device=DEVICE))
        else:
            alpha_loss = torch.tensor(0.).to(DEVICE)

        return policy_loss.item(), alpha_loss.item(), mu.mean().item(), entropy.item()

    def train(self):
        self.training_time_steps += 1

        prioritized_batch_size = int(self.batch_size * self.la3p_prioritized_fraction)
        uniform_batch_size = self.batch_size - prioritized_batch_size

        qf1_losses = []
        qf2_losses = []

        ################################################################
        # UNIFORM BATCH: critic (PAL loss) + actor                     #
        ################################################################
        if uniform_batch_size > 0:
            batch = self.replay_buffer.sample_uniform(uniform_batch_size)
            qf1_loss, qf2_loss = self._train_critic(batch, uniform=True)
            qf1_losses.append(qf1_loss)
            qf2_losses.append(qf2_loss)

            self._train_actor(batch[0])

            self.soft_synchronize_models(
                source_model=self.q_network, target_model=self.target_q_network, tau=self.soft_update_tau
            )

        ################################################################
        # CRITIC-PRIORITIZED BATCH: critic (Huber loss)                #
        ################################################################
        batch = self.replay_buffer.sample_critic(prioritized_batch_size)
        qf1_loss, qf2_loss = self._train_critic(batch, uniform=False)
        qf1_losses.append(qf1_loss)
        qf2_losses.append(qf2_loss)

        self.soft_synchronize_models(
            source_model=self.q_network, target_model=self.target_q_network, tau=self.soft_update_tau
        )

        ################################################################
        # ACTOR-PRIORITIZED BATCH: actor (transitions the critic       #
        # already fits well -- reverse of critic priority)             #
        ################################################################
        actor_observations = self.replay_buffer.sample_actor(prioritized_batch_size)[0]
        policy_loss, alpha_loss, mu, entropy = self._train_actor(actor_observations)

        q_1_td_loss = sum(qf1_losses) / len(qf1_losses)
        q_2_td_loss = sum(qf2_losses) / len(qf2_losses)

        return policy_loss, q_1_td_loss, q_2_td_loss, alpha_loss, mu, entropy

    def soft_synchronize_models(self, source_model, target_model, tau):
        source_model_state = source_model.state_dict()
        target_model_state = target_model.state_dict()
        for k, v in source_model_state.items():
            target_model_state[k] = tau * target_model_state[k] + (1.0 - tau) * v
        target_model.load_state_dict(target_model_state)

    def model_save(self, validation_episode_reward_avg: float) -> None:
        filename = "sac_la3p_{0}_{1:4.1f}_{2}.pth".format(self.env_name, validation_episode_reward_avg, self.current_time)
        torch.save(self.policy.state_dict(), os.path.join(MODEL_DIR, filename))

        copyfile(src=os.path.join(MODEL_DIR, filename), dst=os.path.join(MODEL_DIR, "sac_la3p_{0}_latest.pth".format(self.env_name)))

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

    config = {
        "env_name": ENV_NAME,
        "max_num_episodes": 100_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "steps_between_train": 1,
        "replay_buffer_size": 1_000_000,
        "policy_lr": 7e-4,
        "q_lr": 7e-4,
        "alpha_lr": 7e-4,
        "gamma": 0.99,
        "soft_update_tau": 0.99,
        "validation_time_steps_interval": 30_000,
        "validation_num_episodes": 3,
        "episode_reward_avg_solved": 300,
        "automatic_entropy_tuning": True,
        "print_episode_interval": 10,
        "la3p_alpha": 0.4,                # priority exponent applied to max(|TD-error|, la3p_min_priority)
        "la3p_min_priority": 1.0,         # priority floor; also the Huber-loss transition point
        "la3p_prioritized_fraction": 0.5,  # fraction of batch_size sampled prioritized (vs. uniform) for critic/actor updates
    }

    env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)

    use_wandb = True
    sac = SAC(env=env, test_env=test_env, config=config, use_wandb=use_wandb)
    sac.train_loop()


if __name__ == "__main__":
    main()
