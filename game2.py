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
from search import ElasticModelQuotes, ElasticModelWiki

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

checkpoint = 'checkpoint_000011/checkpoint-11'


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


class GameEnv2(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.cur_pos = 0
        self.action_space = Discrete(5)  # 0 answers, 1 take money, 2-4 help
        self.observation_space = Tuple(
            [
                Box(100, 1_000_000, shape=(1,)),
                # Box(100, 1_000_000, shape=(1,)),
                Box(0, 32_000, shape=(1,)),
                Discrete(2),
                Discrete(2),
                Discrete(2),
                Box(0, 1, shape=(1,)),
                Box(0, 1, shape=(1,)),
                Box(0, 1, shape=(1,)),
                Box(0, 1, shape=(1,)),
            ])

        self.reward_range = (-10_000, 1_000_000)
        # Set the seed. This is only used for the final (reach goal) reward.
        # self.seed(config.worker_index * config.num_workers)

        self.questions = get_one_question()
        self.model = ElasticModelWiki()

        self.question_number = 0
        self.previous_correct = None

        self._trainer = None
        self.scores = []

        self.avail_helps = [1, 1, 1]
        self.obs = None

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
        # action = self._fix_action(action, self.scores)
        if isinstance(action, tuple):
            action = action[0]
        print(f'Question num {self.question_number}', 'action ', action)

        if action == 1:
            print('Taking money')
            return (
                [
                    [money[self.question_number]],
                    [get_fixed(self.question_number)],
                    self.avail_helps[0],
                    self.avail_helps[1],
                    self.avail_helps[2],
                    [0], [0], [0], [0],
                ],
                get_reward(self.question_number) if self.question_number else
                -100,
                True,
                {}
            )

        if action == 2 and self.avail_helps[0]:
            print('next q')
            self.obs, correct = self._get_observation()
            self.previous_correct = correct
            self.avail_helps[0] = 0

            reward = get_fixed(self.question_number)
            return (
                self.obs,
                reward,
                False,
                {}
            )

        if action == 3 and self.avail_helps[1]:
            print('5050')
            self.avail_helps[1] = 0
            second_question = (self.previous_correct + 1) % 4
            self.obs = [
                self.obs[0],
                self.obs[1],
                self.avail_helps[0],
                self.avail_helps[1],
                self.avail_helps[2],
                (self.obs[
                     5] if self.previous_correct == 0 or second_question == 0
                 else [0]),
                (self.obs[
                     6] if self.previous_correct == 1 or second_question == 1
                 else [0]),
                (self.obs[
                     7] if self.previous_correct == 2 or second_question == 2
                 else [0]),
                (self.obs[
                     8] if self.previous_correct == 3 or second_question == 3
                 else [0]),
            ]

            reward = get_fixed(self.question_number)
            return (
                self.obs,
                reward,
                False,
                {}
            )

        if action == 4 and self.avail_helps[2]:
            print('cm')
            self.avail_helps[2] = 0
            scores = self.scores

            best = max(enumerate(scores),
                       key=lambda x: x[1])[0]

            if best == self.previous_correct:
                action = best
            else:
                self.obs = [
                    self.obs[0],
                    self.obs[1],
                    self.avail_helps[0],
                    self.avail_helps[1],
                    self.avail_helps[2],
                    (self.obs[5] if best != 0 else [0]),
                    (self.obs[6] if best != 1 else [0]),
                    (self.obs[7] if best != 2 else [0]),
                    (self.obs[8] if best != 3 else [0]),
                ]

                reward = get_fixed(self.question_number)
                return (
                    self.obs,
                    reward,
                    False,
                    {}
                )

        if action == 2 and not self.avail_helps[0]:
            print('next wrong')
            return (
                [
                    [money[self.question_number]],
                    [get_fixed(self.question_number)],
                    self.avail_helps[0],
                    self.avail_helps[1],
                    self.avail_helps[2],
                    [0], [0], [0], [0],
                ],
                -10_0,
                True,
                {}
            )

        if action == 3 and not self.avail_helps[1]:
            print('5050 wrong')
            return (
                [
                    [money[self.question_number]],
                    [get_fixed(self.question_number)],
                    self.avail_helps[0],
                    self.avail_helps[1],
                    self.avail_helps[2],
                    [0], [0], [0], [0],
                ],
                -10_0,
                True,
                {}
            )

        if action == 4 and not self.avail_helps[2]:
            print('cm wrong')
            return (
                [
                    [money[self.question_number]],
                    [get_fixed(self.question_number)],
                    self.avail_helps[0],
                    self.avail_helps[1],
                    self.avail_helps[2],
                    [0], [0], [0], [0],
                ],
                -100,
                True,
                {}
            )

        action = max(enumerate(self.scores),
                     key=lambda x: x[1])[0]

        if (self.previous_correct is not None
                and self.previous_correct != action):
            print('Wrong answer')
            return (
                [
                    [money[self.question_number]],
                    [get_fixed(self.question_number)],
                    self.avail_helps[0],
                    self.avail_helps[1],
                    self.avail_helps[2],
                    [0], [0], [0], [0],
                ],
                get_fixed(self.question_number),
                True,
                {}
            )
        print('Correct answer')

        done = self.question_number >= len(money)
        reward = (get_reward(self.question_number)
                  if done
                  else get_fixed(self.question_number))

        self.question_number += 1
        self.obs, correct = self._get_observation()
        self.previous_correct = correct

        # print('new obs', (
        #     obs,
        #     reward,
        #     done,
        #     {}
        # ))
        return (
            self.obs,
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
        self.obs, self.previous_correct = self._get_observation()
        # self.previous_correct = None
        self.avail_helps = [1, 1, 1]

        return self.obs

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
                   self.avail_helps[0],
                   self.avail_helps[1],
                   self.avail_helps[2],
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

    def get_action(self, question, answers, q_sum,
                   help_1, help_2, help_3):
        scores = self.model.get_scores(question, answers)
        obs = [
            [q_sum],
            [get_fixed_sum(q_sum)],
            int(help_1),
            int(help_2),
            int(help_3),
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
            action = max(enumerate(scores),
                         key=lambda x: x[1])[0]
        if a == 1:
            action = 'take money'
        if a == 2:
            action = 'new question'
        if a == 3:
            action = 'fifty fifty'
        if a == 4:
            action = 'can mistake'

        return action


config = {
    "env": GameEnv2,
    "env_config": {
    },
    "num_workers": 1,
}

if __name__ == '__main__':

    import logging

    logging.captureWarnings(True)

    tune.run(PPOTrainer,
             config={"env": GameEnv2, "num_workers": 1},
             # stop={"training_iteration": 1},
             checkpoint_freq=1,
             )
