# https://gymnasium.farama.org/environments/classic_control/pendulum/
import copy
import os
import time
from datetime import datetime
from shutil import copyfile

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim import Adam

from a_shared_adam import SharedAdam
from b_actor_and_critic import MODEL_DIR, Actor, Buffer, Critic, Transition


def master_loop(global_actor, shared_stat, run_wandb, global_lock, config):
    env_name = config["env_name"]
    test_env = gym.make(env_name)

    class PPOMaster:
        def __init__(self, global_actor, shared_stat, run_wandb, test_env, global_lock, config):
            self.env_name = config["env_name"]
            self.is_terminated = False

            self.global_actor = global_actor

            self.shared_stat = shared_stat
            self.wandb = run_wandb
            self.test_env = test_env
            self.global_lock = global_lock

            self.validation_episodes_interval = config["validation_episodes_interval"]
            self.validation_num_episodes = config["validation_num_episodes"]
            self.episode_reward_avg_solved = config["episode_reward_avg_solved"]

            self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

            self.last_global_episode_for_validation = 0
            self.last_global_episode_wandb_log = 0

        def validate_loop(self):
            total_train_start_time = time.time()

            while True:
                validation_conditions = [
                    self.shared_stat.global_episodes.value != 0,
                    self.shared_stat.global_episodes.value % self.validation_episodes_interval == 0,
                    self.shared_stat.global_episodes.value > self.last_global_episode_for_validation
                ]
                if all(validation_conditions):
                    self.last_global_episode_for_validation = self.shared_stat.global_episodes.value

                    self.global_lock.acquire()
                    validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

                    total_training_time = time.time() - total_train_start_time
                    total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))

                    print(
                        "[Validation Episode Reward: {0}] Average: {1:.3f}, Elapsed Time: {2}".format(
                            validation_episode_reward_lst, validation_episode_reward_avg, total_training_time
                        )
                    )

                    if validation_episode_reward_avg > self.episode_reward_avg_solved:
                        print(
                            "Solved in {0:,} time steps ({1:,} training steps)!".format(
                                self.shared_stat.global_time_steps.value, self.shared_stat.global_training_time_steps.value
                            )
                        )
                        self.model_save(validation_episode_reward_avg)
                        self.shared_stat.is_terminated.value = 1  # break

                    self.global_lock.release()

                wandb_log_conditions = [
                    self.wandb,
                    self.shared_stat.global_episodes.value > self.last_global_episode_wandb_log,
                    self.shared_stat.global_episodes.value > self.validation_episodes_interval,
                ]
                if all(wandb_log_conditions):
                    self.log_wandb(validation_episode_reward_avg)
                    self.last_global_episode_wandb_log = self.shared_stat.global_episodes.value

                if bool(self.shared_stat.is_terminated.value):
                    if self.wandb:
                        for _ in range(5):
                            self.log_wandb(validation_episode_reward_avg)
                    break

        def validate(self) -> tuple[np.ndarray, float]:
            episode_rewards = np.zeros(self.validation_num_episodes)

            for i in range(self.validation_num_episodes):
                observation, _ = self.test_env.reset()
                episode_reward = 0

                done = False

                while not done:
                    action = self.global_actor.get_action(observation, exploration=False)
                    next_observation, reward, terminated, truncated, _ = self.test_env.step(action * 2)
                    episode_reward += reward
                    observation = next_observation
                    done = terminated or truncated

                episode_rewards[i] = episode_reward

            return episode_rewards, np.average(episode_rewards)

        def log_wandb(self, validation_episode_reward_avg: float) -> None:
            self.wandb.log(
                {
                    "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(
                        self.validation_num_episodes
                    ): validation_episode_reward_avg,
                    "[TRAIN] Episode Reward": self.shared_stat.last_episode_reward.value,
                    "[TRAIN] Policy Loss": self.shared_stat.last_policy_loss.value,
                    "[TRAIN] Critic Loss": self.shared_stat.last_critic_loss.value,
                    "[TRAIN] mu_v": self.shared_stat.last_avg_mu_v.value,
                    "[TRAIN] avg_std_v": self.shared_stat.last_avg_std_v.value,
                    "[TRAIN] avg_action": self.shared_stat.last_avg_action.value,
                    "Training Episode": self.shared_stat.global_episodes.value,
                    "Training Steps": self.shared_stat.global_training_time_steps.value,
                }
            )

        def model_save(self, validation_episode_reward_avg: float) -> None:
            filename = "ppo_{0}_{1:4.1f}_{2}.pth".format(self.env_name, validation_episode_reward_avg, self.current_time)
            torch.save(self.global_actor.state_dict(), os.path.join(MODEL_DIR, filename))

            copyfile(
                src=os.path.join(MODEL_DIR, filename), dst=os.path.join(MODEL_DIR, "ppo_{0}_latest.pth".format(self.env_name))
            )

    master = PPOMaster(
        global_actor=global_actor,
        shared_stat=shared_stat,
        run_wandb=run_wandb,
        test_env=test_env,
        global_lock=global_lock,
        config=config,
    )

    master.validate_loop()

def worker_loop(
        process_id, global_actor, global_critic, global_actor_optimizer, global_critic_optimizer, shared_stat, global_lock, config):
    env_name = config["env_name"]
    env = gym.make(env_name)

    class PPOAgent:
        def __init__(
            self,
            worker_id,
            global_actor,
            global_critic,
            global_actor_optimizer,
            global_critic_optimizer,
            shared_stat,
            env,
            global_lock,
            config
        ):
            self.worker_id = worker_id
            self.env_name = config["env_name"]
            self.env = env
            self.is_terminated = False

            self.global_actor = global_actor
            self.global_critic = global_critic

            self.local_actor = copy.deepcopy(global_actor)
            self.local_actor.load_state_dict(global_actor.state_dict())

            self.local_critic = copy.deepcopy(global_critic)
            self.local_critic.load_state_dict(global_critic.state_dict())

            self.local_actor_optimizer = Adam(self.local_actor.parameters(), lr=config["learning_rate"])
            self.local_critic_optimizer = Adam(self.local_critic.parameters(), lr=config["learning_rate"])

            self.global_actor_optimizer = global_actor_optimizer
            self.global_critic_optimizer = global_critic_optimizer

            self.global_lock = global_lock

            self.max_num_episodes = config["max_num_episodes"]
            self.ppo_epochs = config["ppo_epochs"]
            self.ppo_clip_coefficient = config["ppo_clip_coefficient"]

            self.batch_size = config["batch_size"]
            self.learning_rate = config["learning_rate"]
            self.gamma = config["gamma"]
            self.entropy_beta = config["entropy_beta"]
            self.print_episode_interval = config["print_episode_interval"]

            self.buffer = Buffer()

            self.shared_stat = shared_stat
            self.time_steps = 0
            self.training_time_steps = 0

            self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        def train_loop(self) -> None:
            policy_loss = critic_loss = 0.0

            for n_episode in range(1, self.max_num_episodes + 1):
                episode_reward = 0

                observation, _ = self.env.reset()
                done = False

                while not done:
                    self.time_steps += 1
                    self.shared_stat.global_time_steps.value += 1

                    action = self.local_actor.get_action(observation)
                    next_observation, reward, terminated, truncated, _ = self.env.step(action * 2)

                    episode_reward += reward

                    transition = Transition(observation, action, next_observation, reward, terminated)

                    self.buffer.append(transition)

                    observation = next_observation
                    done = terminated or truncated

                    if self.time_steps % self.batch_size == 0:
                        policy_loss, critic_loss, mu_v, avg_std_v, avg_action = self.train()
                        self.shared_stat.last_policy_loss.value = policy_loss
                        self.shared_stat.last_critic_loss.value = critic_loss
                        self.shared_stat.last_avg_mu_v.value = mu_v
                        self.shared_stat.last_avg_std_v.value = avg_std_v
                        self.shared_stat.last_avg_action.value = avg_action

                        self.buffer.clear()

                self.shared_stat.global_episodes.value += 1
                self.shared_stat.last_episode_reward.value = episode_reward

                if n_episode % self.print_episode_interval == 0:
                    print(
                        "[Worker: {:2}, Episode {:3,}, Time Steps {:6,}]".format(self.worker_id, n_episode, self.time_steps),
                        "Episode Reward: {:>9.3f},".format(episode_reward),
                        "Actor Loss: {:>7.3f},".format(policy_loss),
                        "Critic Loss: {:>7.3f},".format(critic_loss),
                        "Training Steps: {:5,},".format(self.training_time_steps),
                    )

                if bool(self.shared_stat.is_terminated.value):
                    break

        def train(self) -> tuple[float, float, float, float, float]:
            self.training_time_steps += 1
            self.shared_stat.global_training_time_steps.value += 1

            # Getting values from buffer
            observations, actions, next_observations, rewards, dones = self.buffer.get()

            with self.global_lock:
                self.local_critic.load_state_dict(self.global_critic.state_dict())
                self.local_actor.load_state_dict(self.global_actor.state_dict())

                # Resetting gradients of global optimizers
                self.global_actor_optimizer.zero_grad()
                for local_param in self.local_actor.parameters():
                    local_param.grad = None

                self.global_critic_optimizer.zero_grad()
                for local_param in self.local_critic.parameters():
                    local_param.grad = None

            # Save initial local parameters (using deepcopy)
            initial_local_critic_params = copy.deepcopy(self.local_critic.state_dict())
            initial_local_actor_params = copy.deepcopy(self.local_actor.state_dict())


            values = self.local_critic(observations).squeeze(dim=-1)
            next_values = self.local_critic(next_observations).squeeze(dim=-1)
            next_values[dones] = 0.0
            target_values = rewards.squeeze(dim=-1) + self.gamma * next_values
            # Normalized advantage calculation
            advantages = target_values - values
            advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-7)

            old_mu, old_std = self.local_actor.forward(observations)
            old_dist = Normal(old_mu, old_std)
            old_action_log_probs = old_dist.log_prob(value=actions).squeeze(dim=-1)

            for _ in range(self.ppo_epochs):
                values = self.local_critic(observations).squeeze(dim=-1)

                # CRITIC UPDATE
                critic_loss = F.mse_loss(target_values.detach(), values)

                self.local_critic_optimizer.zero_grad()
                critic_loss.backward()
                self.local_critic_optimizer.step()

                # Actor Loss computing
                mu, std = self.local_actor.forward(observations)
                dist = Normal(mu, std)
                action_log_probs = dist.log_prob(value=actions).squeeze(dim=-1)

                ratio = torch.exp(action_log_probs - old_action_log_probs.detach())

                ratio_advantages = ratio * advantages.detach()
                clipped_ratio_advantages = (
                    torch.clamp(ratio, 1 - self.ppo_clip_coefficient, 1 + self.ppo_clip_coefficient) * advantages.detach()
                )
                ratio_advantages_sum = torch.min(ratio_advantages, clipped_ratio_advantages).sum()

                entropy = dist.entropy().squeeze(dim=-1)
                entropy_sum = entropy.sum()

                actor_loss = -1.0 * ratio_advantages_sum - 1.0 * entropy_sum * self.entropy_beta

                # Actor Update
                self.local_actor_optimizer.zero_grad()
                actor_loss.backward()
                self.local_actor_optimizer.step()

            # Calculate the difference between updated and initial local parameters
            delta_local_critic_grads = {
                name: (initial_local_critic_params[name] - self.local_critic.state_dict()[name]) / self.learning_rate
                for name in self.local_critic.state_dict()}

            delta_local_actor_grads = {
                name: (initial_local_actor_params[name] - self.local_actor.state_dict()[name]) / self.learning_rate
                for name in self.local_actor.state_dict()}

            # Updating global model parameters
            with self.global_lock:
                self.global_critic_optimizer.zero_grad()
                for name, global_param in self.global_critic.named_parameters():
                    global_param.grad = delta_local_critic_grads[name]
                self.global_critic_optimizer.step()

                self.global_actor_optimizer.zero_grad()
                for name, global_param in self.global_actor.named_parameters():
                    global_param.grad = delta_local_actor_grads[name]
                self.global_actor_optimizer.step()

            return (
                actor_loss.item(),
                critic_loss.item(),
                mu.mean().item(),
                std.mean().item(),
                actions.mean().item(),
            )

    agent = PPOAgent(
        worker_id=process_id,
        global_actor=global_actor,
        global_critic=global_critic,
        global_actor_optimizer=global_actor_optimizer,
        global_critic_optimizer=global_critic_optimizer,
        shared_stat=shared_stat,
        env=env,
        global_lock=global_lock,
        config=config,
    )

    agent.train_loop()


class SharedStat:
    def __init__(self):
        self.global_episodes = mp.Value("I", 0)  # I: unsigned int
        self.global_time_steps = mp.Value("I", 0)  # I: unsigned int
        self.global_training_time_steps = mp.Value("I", 0)  # I: unsigned int

        self.last_episode_reward = mp.Value("d", 0.0)  # d: double
        self.last_policy_loss = mp.Value("d", 0.0)  # d: double
        self.last_critic_loss = mp.Value("d", 0.0)  # d: double
        self.last_avg_mu_v = mp.Value("d", 0.0)  # d: double
        self.last_avg_std_v = mp.Value("d", 0.0)  # d: double
        self.last_avg_action = mp.Value("d", 0.0)  # d: double

        self.is_terminated = mp.Value("I", 0)  # I: unsigned int --> bool


class PPO:
    def __init__(self, config: dict, use_wandb: bool):
        self.config = config
        self.use_wandb = use_wandb
        self.num_workers = min(config["num_workers"], mp.cpu_count() - 1)

        # Initialize global models and optimizers
        self.global_actor = Actor(n_features=3, n_actions=1).share_memory()
        self.global_critic = Critic(n_features=3).share_memory()

        self.global_actor_optimizer = SharedAdam(self.global_actor.parameters(), lr=config["learning_rate"])
        self.global_critic_optimizer = SharedAdam(self.global_critic.parameters(), lr=config["learning_rate"])

        self.global_lock = mp.Lock()
        self.shared_stat = SharedStat()

        if use_wandb:
            current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
            self.wandb = wandb.init(project="PPO_{0}".format(config["env_name"]), name=current_time, config=config)
        else:
            self.wandb = None

        self.worker_processes = []
        self.master_process = None

    def train_loop(self) -> None:
        for i in range(self.num_workers):
            worker_process = mp.Process(
                target=worker_loop,
                args=(
                    i,
                    self.global_actor,
                    self.global_critic,
                    self.global_actor_optimizer,
                    self.global_critic_optimizer,
                    self.shared_stat,
                    self.global_lock,
                    self.config,
                ),
            )
            worker_process.start()
            print(">>> Worker Process: {0} Started!".format(worker_process.pid))
            self.worker_processes.append(worker_process)

        master_process = mp.Process(
            target=master_loop, args=(self.global_actor, self.shared_stat, self.wandb, self.global_lock, self.config)
        )
        master_process.start()
        print(">>> Master Process: {0} Started!".format(master_process.pid))

        ###########################

        for worker_process in self.worker_processes:
            worker_process.join()
            print(">>> Worker Process: {0} Joined!".format(worker_process.pid))

        master_process.join()
        print(">>> Master Process: {0} Joined!".format(master_process.pid))

        if self.use_wandb:
            self.wandb.finish()


def main() -> None:
    print("TORCH VERSION:", torch.__version__)
    ENV_NAME = "Pendulum-v1"

    config = {
        "env_name": ENV_NAME,                               # 환경의 이름
        "num_workers": 1,                                   # 동시 수행 Worker Process 수
        "max_num_episodes": 200_000,                        # 훈련을 위한 최대 에피소드 횟수
        "ppo_epochs": 10,                                   # PPO 내부 업데이트 횟수
        "ppo_clip_coefficient": 0.2,                        # PPO Ratio Clip Coefficient
        "batch_size": 256,                                  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "learning_rate": 0.0003,                            # 학습율
        "gamma": 0.99,                                      # 감가율
        "entropy_beta": 0.03,                               # 엔트로피 가중치
        "print_episode_interval": 20,                       # Episode 통계 출력에 관한 에피소드 간격
        "validation_episodes_interval": 100,   # 검증 사이 마다 각 훈련 time steps 간격
        "validation_num_episodes": 3,                       # 검증에 수행하는 에피소드 횟수
        "episode_reward_avg_solved": -100,                  # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
    }

    use_wandb = True
    ppo = PPO(use_wandb=use_wandb, config=config)
    ppo.train_loop()


if __name__ == "__main__":
    main()
