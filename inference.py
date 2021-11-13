"""
Example showing how you can use your trained policy for inference
(computing actions) in an environment.
Includes options for LSTM-based models (--use-lstm), attention-net models
(--use-attention), and plain (non-recurrent) models.
"""
import argparse
import gym
import os

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.registry import get_trainer_class

from game import GameEnv

checkpoint = '/home/elizaveta/ray_results/PPO_2021-11-13_01-13-53' \
             '/PPO_GameEnv_d2655_00000_0_2021-11-13_01-13-54/checkpoint_000001/checkpoint-1'


ray.init(num_cpus=12)

config = {
    "env": GameEnv,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
}


# Create new Trainer and restore its state from the last checkpoint.
trainer = PPOTrainer(config=config)
trainer.restore(checkpoint)

# Create the env to do inference in.
# env = gym.make(GameEnv)
# obs = env.reset()

num_episodes = 0
episode_reward = 0.0

while num_episodes < 1:
    # Compute an action (`a`).
    a = trainer.compute_single_action(
        observation=[
                    [100],
                    # [0],
                    # 1, 1, 1,
                    [10], [10], [10], [10],
                ],
        explore=False,
        policy_id="default_policy",  # <- default value
    )
    print(a)
    # Send the computed action `a` to the env.
    # obs, reward, done, _ = env.step(a)
    # episode_reward += reward
    # Is the episode `done`? -5> Reset.
    # if done:
    #     print(f"Episode done: Total reward = {episode_reward}")
    #     obs = env.reset()
    #     num_episodes += 1
    #     episode_reward = 0.0

if __name__ == '__main__':

    ray.shutdown()