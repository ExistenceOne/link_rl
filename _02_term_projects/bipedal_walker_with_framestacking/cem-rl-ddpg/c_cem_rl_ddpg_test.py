import os
import sys

import gymnasium as gym
import numpy as np
import torch

from a_ddpg_models import MODEL_DIR, Actor


def make_env(env_name: str, stack_size: int, render_mode: str = None) -> gym.Env:
    env = gym.make(env_name, render_mode=render_mode)
    if stack_size and stack_size > 1:
        env = gym.wrappers.FrameStackObservation(env, stack_size=stack_size)
    return env


def test(env: gym.Env, actor: Actor, num_episodes: int) -> None:
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        observation, _ = env.reset()

        episode_steps = 0

        done = False

        while not done:
            episode_steps += 1
            action = actor.get_action(observation, exploration=False)

            next_observation, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(i, episode_steps, episode_reward))


def main_play(num_episodes: int, env_name: str, stack_size: int, model_filename: str) -> None:
    env = make_env(env_name, stack_size, render_mode="human")

    obs_shape = env.observation_space.shape
    n_features = int(np.prod(obs_shape))
    obs_ndim = len(obs_shape)
    n_actions = env.action_space.shape[0]

    actor = Actor(n_features=n_features, n_actions=n_actions, obs_ndim=obs_ndim)
    model_params = torch.load(os.path.join(MODEL_DIR, model_filename), weights_only=True)
    actor.load_state_dict(model_params)
    actor.eval()

    test(env, actor, num_episodes=num_episodes)

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 3
    ENV_NAME = "BipedalWalkerHardcore-v3"
    STACK_SIZE = 1  # must match the stack_size used during training

    if len(sys.argv) > 1:
        MODEL_FILENAME = sys.argv[1]
    else:
        MODEL_FILENAME = "cem_rl_ddpg_{0}_latest.pth".format(ENV_NAME)

    main_play(num_episodes=NUM_EPISODES, env_name=ENV_NAME, stack_size=STACK_SIZE, model_filename=MODEL_FILENAME)
