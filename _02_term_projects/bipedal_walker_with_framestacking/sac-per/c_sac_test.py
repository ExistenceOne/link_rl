import os
import sys

import gymnasium as gym
import torch

from a_sac_models import MODEL_DIR, GaussianPolicy


def test(env: gym.Env, actor: GaussianPolicy, num_episodes: int) -> None:
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


def main_play(num_episodes: int, env_name: str, model_filename: str) -> None:
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env=env, video_folder="videos",
        name_prefix="sac_per_{0}".format(model_filename)
    )

    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    policy = GaussianPolicy(n_features=n_features, n_actions=n_actions, action_space=env.action_space)
    model_params = torch.load(os.path.join(MODEL_DIR, model_filename), weights_only=True)
    policy.load_state_dict(model_params)
    policy.eval()

    test(env, policy, num_episodes=num_episodes)
    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 2
    ENV_NAME = "BipedalWalkerHardcore-v3"

    if len(sys.argv) > 1:
        MODEL_FILENAME = sys.argv[1]
    else:
        MODEL_FILENAME = "sac_per_{0}_latest.pth".format(ENV_NAME)

    main_play(num_episodes=NUM_EPISODES, env_name=ENV_NAME, model_filename=MODEL_FILENAME)
