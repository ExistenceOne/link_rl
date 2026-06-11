# https://gymnasium.farama.org/environments/classic_control/pendulum/
import os
import time
from datetime import datetime
from shutil import copyfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import wandb
from a_actor_and_critic import MODEL_DIR, Actor, Buffer, Critic, Transition


class PPO:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]

        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if use_wandb:
            self.wandb = wandb.init(project="PPO_{0}_with_framestacking".format(self.env_name), name=self.current_time, config=config)
        else:
            self.wandb = None

        self.max_num_episodes = config["max_num_episodes"]
        self.ppo_epochs = config["ppo_epochs"]
        self.ppo_clip_coefficient = config["ppo_clip_coefficient"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.entropy_beta = config["entropy_beta"]
        self.max_grad_norm = config["max_grad_norm"]
        self.print_episode_interval = config["print_episode_interval"]
        self.validation_time_steps_interval = config["validation_time_steps_interval"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_save = config["episode_reward_avg_save"]

        n_features = int(np.prod(env.observation_space.shape))  # (4,24) -> 96
        n_actions = env.action_space.shape[0]

        self.actor = Actor(n_features=n_features, n_actions=n_actions)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic = Critic(n_features=n_features)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.buffer = Buffer()

        self.time_steps = 0
        self.training_time_steps = 0

    def train_loop(self) -> None:
        total_train_start_time = time.time()

        validation_episode_reward_avg = -200
        actor_loss = critic_loss = mu_v = avg_std_v = avg_action = 0.0
        approx_kl = clip_frac = entropy = action_clip_frac = 0.0

        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            episode_reward = 0
            episode_action_clip_frac_sum = 0.0
            episode_action_clip_frac_count = 0

            observation, _ = self.env.reset()
            done = False

            while not done:
                self.time_steps += 1

                action, sampled_action_clip_frac = self.actor.get_action(observation, return_clip_frac=True)
                episode_action_clip_frac_sum += sampled_action_clip_frac
                episode_action_clip_frac_count += 1

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)
                self.buffer.append(transition)

                observation = next_observation
                done = terminated or truncated

                if self.time_steps % self.batch_size == 0:
                    actor_loss, critic_loss, mu_v, avg_std_v, avg_action, approx_kl, clip_frac, entropy, _ = self.train()
                    self.buffer.clear()

                if self.time_steps % self.validation_time_steps_interval == 0:
                    validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

                    total_training_time = time.time() - total_train_start_time
                    total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))

                    print(
                        "[Validation Episode Reward: {0}] Average: {1:.3f}, Elapsed Time: {2}".format(
                            validation_episode_reward_lst, validation_episode_reward_avg, total_training_time
                        )
                    )

                    if validation_episode_reward_avg > self.episode_reward_avg_save:
                        print("Solved in {0:,} time steps ({1:,} training steps)!".format(self.time_steps, self.training_time_steps))
                        self.model_save(validation_episode_reward_avg)
                        # is_terminated = True

                    if self.use_wandb:
                        self.log_wandb(
                            validation_episode_reward_avg,
                            episode_reward,
                            actor_loss,
                            critic_loss,
                            mu_v,
                            avg_std_v,
                            avg_action,
                            n_episode,
                        )

            if n_episode % self.print_episode_interval == 0:
                action_clip_frac = episode_action_clip_frac_sum / max(1, episode_action_clip_frac_count)
                print(
                    "[Episode {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "Episode Reward: {:>9.3f},".format(episode_reward),
                    "Actor Loss: {:>7.3f},".format(actor_loss),
                    "Critic Loss: {:>7.3f},".format(critic_loss),
                    "Approx KL: {:>7.4f},".format(approx_kl),
                    "Clip Frac: {:>7.4f},".format(clip_frac),
                    "Entropy: {:>7.4f},".format(entropy),
                    "Action Clip Frac: {:>7.4f},".format(action_clip_frac),
                    "Training Steps: {:5,}, ".format(self.training_time_steps),
                )

            if is_terminated:
                if self.wandb:
                    for _ in range(5):
                        self.log_wandb(
                            validation_episode_reward_avg,
                            episode_reward,
                            actor_loss,
                            critic_loss,
                            mu_v,
                            avg_std_v,
                            avg_action,
                            n_episode,
                        )
                break

        total_training_time = time.time() - total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))
        if self.use_wandb:
            self.wandb.finish()

    def log_wandb(
        self,
        validation_episode_reward_avg: float,
        episode_reward: float,
        actor_loss: float,
        critic_loss: float,
        mu_v: float,
        avg_std_v: float,
        avg_action: float,
        n_episode: float,
    ) -> None:
        self.wandb.log(
            {
                "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(
                    self.validation_num_episodes
                ): validation_episode_reward_avg,
                "[TRAIN] Episode Reward": episode_reward,
                "[TRAIN] Actor Loss": actor_loss,
                "[TRAIN] Critic Loss": critic_loss,
                "[TRAIN] mu_v": mu_v,
                "[TRAIN] avg_std_v": avg_std_v,
                "[TRAIN] avg_action": avg_action,
                "Training Episode": n_episode,
                "Training Steps": self.training_time_steps,
            }
        )

    def train(self) -> tuple[float, float, float, float, float, float, float, float, float]:
        self.training_time_steps += 1

        observations, actions, next_observations, rewards, dones = self.buffer.get()

        # GAE computation with no gradient
        with torch.no_grad():
            values = self.critic(observations).squeeze(dim=-1)
            next_values = self.critic(next_observations).squeeze(dim=-1)
            next_values[dones] = 0.0

            deltas = rewards.squeeze(dim=-1) + self.gamma * next_values - values
            not_done = (~dones).float()

            advantages = torch.zeros_like(deltas)
            gae = 0.0
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + self.gamma * self.gae_lambda * not_done[t] * gae
                advantages[t] = gae

            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Compute old log probs before any parameter updates
        old_mu, old_std = self.actor.forward(observations)
        old_dist = Normal(old_mu, old_std)
        old_action_log_probs = old_dist.log_prob(value=actions).sum(dim=-1).detach()
        action_clip_frac = ((actions < -1.0) | (actions > 1.0)).float().mean().item()

        for _ in range(self.ppo_epochs):
            # CRITIC UPDATE
            values = self.critic(observations).squeeze(dim=-1)
            critic_loss = F.mse_loss(returns.detach(), values)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            # ACTOR UPDATE with clipped PPO objective
            mu, std = self.actor.forward(observations)
            dist = Normal(mu, std)
            action_log_probs = dist.log_prob(value=actions).sum(dim=-1)

            ratio = torch.exp(action_log_probs - old_action_log_probs)

            ratio_advantages = ratio * advantages.detach()
            clipped_ratio_advantages = (
                torch.clamp(ratio, 1 - self.ppo_clip_coefficient, 1 + self.ppo_clip_coefficient) * advantages.detach()
            )
            ratio_advantages = torch.min(ratio_advantages, clipped_ratio_advantages).mean()

            entropy = dist.entropy().sum(dim=-1).mean()

            actor_loss = -1.0 * ratio_advantages - self.entropy_beta * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

        mu, std = self.actor.forward(observations)
        dist = Normal(mu, std)
        action_log_probs = dist.log_prob(value=actions).sum(dim=-1)

        with torch.no_grad():
            log_ratio = action_log_probs - old_action_log_probs
            ratio = torch.exp(log_ratio)

            approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
            clip_frac = ((ratio - 1.0).abs() > self.ppo_clip_coefficient).float().mean().item()

        entropy = dist.entropy().mean().item()

        return (
            actor_loss.item(),
            critic_loss.item(),
            mu.mean().item(),
            std.mean().item(),
            actions.mean().item(),
            approx_kl,
            clip_frac,
            entropy,
            action_clip_frac,
        )

    def model_save(self, validation_episode_reward_avg: float) -> None:
        filename = "ppo_{0}_{1:4.1f}_{2}.pth".format(self.env_name, validation_episode_reward_avg, self.current_time)
        torch.save(self.actor.state_dict(), os.path.join(MODEL_DIR, filename))

        copyfile(src=os.path.join(MODEL_DIR, filename), dst=os.path.join(MODEL_DIR, "ppo_{0}_latest.pth".format(self.env_name)))

    def validate(self) -> tuple[np.ndarray, float]:
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        for i in range(self.validation_num_episodes):
            episode_reward = 0

            observation, _ = self.test_env.reset()
            done = False

            while not done:
                action = self.actor.get_action(observation, exploration=False)

                next_observation, reward, terminated, truncated, _ = self.test_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

        return episode_reward_lst, np.average(episode_reward_lst)


def main() -> None:
    print("TORCH VERSION:", torch.__version__)
    ENV_NAME = "BipedalWalkerHardcore-v3"

    env = gym.make(ENV_NAME)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    test_env = gym.make(ENV_NAME)
    test_env = gym.wrappers.FrameStackObservation(test_env, stack_size=4)

    config = {
        "env_name": ENV_NAME,
        "max_num_episodes": 50_000,
        "ppo_epochs": 10,
        "ppo_clip_coefficient": 0.2,                # PPO Ratio Clip Coefficient
        "batch_size": 256,
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,                         # GAE lambda
        "entropy_beta": 0.025,                       # 엔트로피 가중치
        "max_grad_norm": 0.5,                       # Gradient clipping norm
        "print_episode_interval": 10,
        "validation_time_steps_interval": 30_000,
        "validation_num_episodes": 3,
        "episode_reward_avg_save": 0,
    }

    use_wandb = True
    ppo = PPO(env=env, test_env=test_env, config=config, use_wandb=use_wandb)
    ppo.train_loop()


if __name__ == "__main__":
    main()
