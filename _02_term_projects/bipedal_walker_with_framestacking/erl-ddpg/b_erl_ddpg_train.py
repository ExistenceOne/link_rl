# Evolutionary Reinforcement Learning (ERL, Khadka & Tumer 2018) on top of DDPG.
#
# A GA-evolved population of deterministic Actor networks plus one gradient-based DDPG learner share
# a single replay buffer. Population members are evaluated by REAL environment rollouts (fitness =
# episodic return); their transitions fill the buffer, DDPG trains from the buffer, and the DDPG
# actor is periodically injected back into the population.
import os
import time
from datetime import datetime
from shutil import copyfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from a_ddpg_models import MODEL_DIR, Actor, QCritic, ReplayBuffer, DEVICE

import wandb


def make_env(env_name: str, stack_size: int, render_mode: str = None) -> gym.Env:
    env = gym.make(env_name, render_mode=render_mode)
    if stack_size and stack_size > 1:
        env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)
    return env


class ERL_DDPG:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]
        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if use_wandb:
            self.wandb = wandb.init(project="erl_ddpg_{0}".format(self.env_name), name=self.current_time, config=config)
        else:
            self.wandb = None

        self.max_generations = config["max_generations"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.soft_update_tau = config["soft_update_tau"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.learning_starts = config["learning_starts"]
        self.exploration_noise = config["exploration_noise"]
        self.grad_steps_ratio = config["grad_steps_ratio"]

        self.pop_size = config["pop_size"]
        self.n_elite = config["n_elite"]
        self.eval_episodes = config["eval_episodes"]
        self.mut_prob = config["mut_prob"]
        self.mut_strength = config["mut_strength"]
        self.tournament_k = config["tournament_k"]
        self.sync_period = config["sync_period"]

        self.print_generation_interval = config["print_generation_interval"]
        self.validation_generation_interval = config["validation_generation_interval"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_save = config["episode_reward_avg_save"]

        obs_shape = env.observation_space.shape
        self.n_features = int(np.prod(obs_shape))
        self.obs_ndim = len(obs_shape)
        self.n_actions = env.action_space.shape[0]

        # DDPG learner
        self.actor = self._new_actor()
        self.target_actor = self._new_actor()
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.q_critic = QCritic(self.n_features, self.n_actions, obs_ndim=self.obs_ndim)
        self.target_q_critic = QCritic(self.n_features, self.n_actions, obs_ndim=self.obs_ndim)
        self.target_q_critic.load_state_dict(self.q_critic.state_dict())
        self.q_critic_optimizer = optim.Adam(self.q_critic.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(
            capacity=self.replay_buffer_size, observation_shape=obs_shape, n_actions=self.n_actions
        )

        # Population of additional actors evolved by the GA
        self.population = [self._new_actor() for _ in range(self.pop_size)]

        self.time_steps = 0
        self.training_time_steps = 0
        self.total_train_start_time = None

    def _new_actor(self) -> Actor:
        return Actor(
            n_features=self.n_features, n_actions=self.n_actions,
            obs_ndim=self.obs_ndim, exploration_noise=self.exploration_noise,
        )

    # ----------------------------------------------------------------- rollouts
    def rollout(self, actor: Actor, exploration: bool, store: bool) -> tuple[float, int]:
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

    def evaluate_actor(self, actor: Actor, exploration: bool, store: bool) -> tuple[float, int]:
        rewards, total_steps = [], 0
        for _ in range(self.eval_episodes):
            r, s = self.rollout(actor, exploration=exploration, store=store)
            rewards.append(r)
            total_steps += s
        return float(np.mean(rewards)), total_steps

    # ----------------------------------------------------------------- DDPG update
    def ddpg_train_step(self):
        self.training_time_steps += 1
        observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(self.batch_size)

        # CRITIC UPDATE
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

        # ACTOR UPDATE
        actor_loss = -1.0 * self.q_critic(observations, self.actor(observations)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_synchronize_models(self.actor, self.target_actor, self.soft_update_tau)
        self.soft_synchronize_models(self.q_critic, self.target_q_critic, self.soft_update_tau)

        return actor_loss.item(), critic_loss.item()

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

    def _crossover(self, parent_a: Actor, parent_b: Actor) -> dict:
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
        order = list(np.argsort(fitnesses)[::-1])
        new_states = []
        for e in order[: self.n_elite]:
            new_states.append({k: v.clone() for k, v in self.population[e].state_dict().items()})
        while len(new_states) < self.pop_size:
            pa = self.population[self._tournament_select(fitnesses)]
            pb = self.population[self._tournament_select(fitnesses)]
            child = self._crossover(pa, pb)
            self._mutate(child)
            new_states.append(child)
        for actor, st in zip(self.population, new_states):
            actor.load_state_dict(st)

    def inject_rl_actor(self) -> None:
        self.population[-1].load_state_dict(self.actor.state_dict())

    # ----------------------------------------------------------------- main loop
    def train_loop(self) -> None:
        self.total_train_start_time = time.time()
        validation_episode_reward_avg = -1000.0
        actor_loss = critic_loss = 0.0

        for generation in range(1, self.max_generations + 1):
            fitnesses, steps_this_gen = [], 0
            for actor in self.population:
                fitness, steps = self.evaluate_actor(actor, exploration=False, store=True)
                fitnesses.append(fitness)
                steps_this_gen += steps
            self.time_steps += steps_this_gen

            rl_return, rl_steps = self.rollout(self.actor, exploration=True, store=True)
            self.time_steps += rl_steps
            steps_this_gen += rl_steps

            if self.time_steps > self.learning_starts and self.replay_buffer.size() > self.batch_size:
                num_updates = max(1, int(steps_this_gen * self.grad_steps_ratio))
                for _ in range(num_updates):
                    actor_loss, critic_loss = self.ddpg_train_step()

            self.evolve(fitnesses)

            if generation % self.sync_period == 0:
                self.inject_rl_actor()

            best_fitness = float(np.max(fitnesses))
            mean_fitness = float(np.mean(fitnesses))

            if generation % self.validation_generation_interval == 0:
                validation_episode_reward_avg = self.validate()
                if validation_episode_reward_avg > self.episode_reward_avg_save:
                    self.model_save(validation_episode_reward_avg)

            if generation % self.print_generation_interval == 0:
                print(
                    "[Gen {:4,}, Time Steps {:7,}]".format(generation, self.time_steps),
                    "Pop Best: {:>8.2f}, Pop Mean: {:>8.2f},".format(best_fitness, mean_fitness),
                    "RL Return: {:>8.2f},".format(rl_return),
                    "Actor L.: {:>8.3f}, Critic L.: {:>7.3f},".format(actor_loss, critic_loss),
                    "Train Steps: {:6,}".format(self.training_time_steps),
                )

            if self.use_wandb:
                self.log_wandb(generation, validation_episode_reward_avg, best_fitness, mean_fitness,
                               rl_return, actor_loss, critic_loss)

        total_training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.total_train_start_time))
        print("Total Training End : {}".format(total_training_time))
        if self.use_wandb:
            self.wandb.finish()

    def log_wandb(self, generation, validation_episode_reward_avg, best_fitness, mean_fitness,
                  rl_return, actor_loss, critic_loss) -> None:
        self.wandb.log({
            "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(self.validation_num_episodes): validation_episode_reward_avg,
            "[EVO] population best fitness": best_fitness,
            "[EVO] population mean fitness": mean_fitness,
            "[TRAIN] rl actor return": rl_return,
            "[TRAIN] actor loss": actor_loss,
            "[TRAIN] critic loss": critic_loss,
            "[TRAIN] Replay buffer": self.replay_buffer.size(),
            "generation": generation,
            "training steps": self.training_time_steps,
        })

    def model_save(self, validation_episode_reward_avg: float) -> None:
        filename = "erl_ddpg_{0}_{1:4.1f}_{2}.pth".format(self.env_name, validation_episode_reward_avg, self.current_time)
        torch.save(self.actor.state_dict(), os.path.join(MODEL_DIR, filename))
        copyfile(src=os.path.join(MODEL_DIR, filename), dst=os.path.join(MODEL_DIR, "erl_ddpg_{0}_latest.pth".format(self.env_name)))

    def validate(self) -> float:
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)
        for i in range(self.validation_num_episodes):
            observation, _ = self.test_env.reset()
            episode_reward, done = 0.0, False
            while not done:
                action = self.actor.get_action(observation, exploration=False)
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
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "soft_update_tau": 0.995,
        "replay_buffer_size": 1_000_000,
        "learning_starts": 10_000,
        "exploration_noise": 0.1,
        "grad_steps_ratio": 1.0,
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
    agent = ERL_DDPG(env=env, test_env=test_env, config=config, use_wandb=use_wandb)
    agent.train_loop()


if __name__ == "__main__":
    main()
