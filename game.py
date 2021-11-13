import argparse
import gym
from gym.spaces import Box, Discrete, Space, Tuple
import numpy as np
import os
import random

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print

from run_questions import get_one_question
from search import ElasticModelQuotes, ElasticModelSource, ElasticModelWiki

money = [
    100,
    200,
    300,
    500,
    1000,
    2000,
    4000,
    8000,
    16000,
    32000,
    64000,
    125_000,
    250_000,
    500_000,
    1_000_000,
]

checkpoint = '/home/elizaveta/ray_results/PPO_2021-11-13_01-13-53' \
             '/PPO_GameEnv_d2655_00000_0_2021-11-13_01-13-54/checkpoint_000001/checkpoint-1'


def get_reward(q_num):
    return money[q_num]


def get_fixed(q_num):
    if money[q_num] < 1000:
        return 0
    if money[q_num] < 32000:
        return 1000
    return 32000


def get_fixed_sum(q_num):
    if q_num < 1000:
        return 0
    if q_num < 32000:
        return 1000
    return 32000


class GameEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.cur_pos = 0
        self.action_space = Discrete(5)  # 0-3 answers, 4 take money
        self.observation_space = Tuple(
            [
                Box(100, 1_000_000, shape=(1,)),
                # Box(100, 1_000_000, shape=(1,)),
                Box(0, 32_000, shape=(1,)),
                # Discrete(2),
                # Discrete(2),
                # Discrete(2),
                Box(0, 700, shape=(1,)),
                Box(0, 700, shape=(1,)),
                Box(0, 700, shape=(1,)),
                Box(0, 700, shape=(1,)),
            ])  # 0 quest money, 1 fixed
        # money,
        # 2-4 helps, 5 model ans, 6 score

        self.reward_range = (0, 1_000_000)
        # Set the seed. This is only used for the final (reach goal) reward.
        self.seed(config.worker_index * config.num_workers)

        self.questions = get_one_question()
        self.model = ElasticModelSource()

        self.question_number = 0
        self.previous_correct = None

        self._trainer = None
        self.scores = []

    def _fix_action(self, action, scores):
        if action == 4 and self.question_number == 0:
            return max(enumerate(scores), key=lambda x: x[0])
        return action

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        action = int(action)
        action = self._fix_action(action, self.scores)
        # print(f'Question num {self.question_number}', action, type(action))
        # print(action == 4)
        # print(action == 5)

        if action == 4:
            # print('Taking money')
            return (
                [
                    [money[self.question_number]],
                    [get_fixed(self.question_number)],
                    # [0],
                    # 1, 1, 1,
                    [10], [10], [10], [10],
                ],
                get_reward(self.question_number),
                True,
                {}
            )

        if (self.previous_correct is not None
                and self.previous_correct != action):
            # print('Wrong answer')
            return (
                [
                    [money[self.question_number]],
                    [get_fixed(self.question_number)],
                    # [0],
                    # 1, 1, 1,
                    [10], [10], [10], [10],
                ],
                get_fixed(self.question_number),
                True,
                {}
            )
        # print('Correct answer')

        done = self.question_number >= len(money)
        reward = (get_reward(self.question_number)
                  if done
                  else get_fixed(self.question_number))

        self.question_number += 1
        obs, correct = self._get_observation()
        self.previous_correct = correct

        # print('new obs', (
        #     obs,
        #     reward,
        #     done,
        #     {}
        # ))
        return (
            obs,
            reward,
            done,
            {}
        )

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """
        self.question_number = 0
        obs, self.previous_correct = self._get_observation()
        # self.previous_correct = None

        return obs

    def _get_observation(self):
        # 0 quest money, 1 fixed
        # money,
        # 2-4 helps, 5 model ans, 5 score
        try:
            question, answers, correct = self.questions.send(None)
        except StopIteration:
            self.questions = get_one_question()
            question, answers, correct = self.questions.send(None)

        scores = self.model.get_scores(question, answers)
        self.scores = scores

        try:
            next_q = [money[self.question_number]]
        except IndexError:
            next_q = [max(money)]

        # print(question)
        return [
                   next_q,
                   [get_fixed(self.question_number)],
                   # 1,
                   # 1,
                   # 1,
                   [scores[0]],
                   [scores[1]],
                   [scores[2]],
                   [scores[3]],
               ], correct

    @property
    def trainer(self):
        if self._trainer is None:
            self._trainer = PPOTrainer(config=config)
            self._trainer.restore(checkpoint)

        return self._trainer

    def get_action(self, question, answers, q_sum):
        scores = self.model.get_scores(question, answers)
        obs = [
            q_sum,
            [get_fixed_sum(q_sum)],
            1,
            1,
            1,
            [scores[0]],
            [scores[1]],
            [scores[2]],
            [scores[3]],
        ]
        a = self.trainer.compute_single_action(
            observation=obs,
            explore=False,
            policy_id="default_policy",  # <- default value
        )

        if a == 0:
            action = max(enumerate(self.scores),
                         key=lambda x: x[1])[0]
        if a == 1:
            action = 'take money'
        if a == 2:
            action = 'next question'
        if a == 3:
            action = '5050'
        if a == 4:
            action = 'cm'

        return action


if __name__ == '__main__':
    config = {
        "env": GameEnv,
        "env_config": {
        },
        "num_workers": 1,
    }

    import logging

    logging.captureWarnings(True)

    tune.run(PPOTrainer,
             config={"env": GameEnv, "num_workers": 1},
             # stop={"training_iteration": 1},
             checkpoint_freq=1,
             )
