#!/usr/bin/env python

from typing import Dict
import random
import numpy as np
from gym import spaces

from catalyst.rl.core import EnvironmentSpec
from catalyst.rl.utils import extend_space

from environment import wrap_l2m_env, L2M2 as L2M2019Env


BIG_NUM = np.iinfo(np.int32).max


class SkeletonEnvWrapper(EnvironmentSpec):
    def __init__(
        self,
        history_len=1,
        frame_skip=1,
        reward_scale=1,
        step_reward=0,
        action_mean=None,
        action_std=None,
        min_episode_steps=None,
        min_episode_raw_reward=None,
        hard_seed_probability=0.5,
        hard_seed_initial_episode=int(1e3),
        visualize=False,
        mode="train",
        sampler_id=None,
        reload_period=None,
        difficulty_probabilities=None,
        environment_params: Dict = None,
        wrapper_params: Dict = None,
    ):
        super().__init__(visualize=visualize, mode=mode, sampler_id=sampler_id)
        self._environment_params = environment_params or {}
        self._wrapper_params = wrapper_params or {}
        self.env = None

        self._history_len = history_len
        self._frame_skip = frame_skip
        self._reward_scale = reward_scale
        self._step_reward = step_reward
        self._reload_period = reload_period or BIG_NUM
        self._difficulty_probabilities = difficulty_probabilities
        if self._difficulty_probabilities is not None:
            assert np.isclose(1.0, np.sum(self._difficulty_probabilities))

        self.action_mean = np.array(action_mean) \
            if action_mean is not None else None
        self.action_std = np.array(action_std) \
            if action_std is not None else None

        self._prepare_env()
        self._prepare_spaces()

        self._episode = 0
        self._episode_step = 0
        self._episode_reward = 0
        self._episode_seed = None
        self._min_episode_steps = min_episode_steps or -np.inf
        self._min_episode_raw_reward = min_episode_raw_reward or -np.inf
        self._hard_seeds = set()
        self._hard_seed_probability = hard_seed_probability
        self._hard_seed_initial_episode = hard_seed_initial_episode

    @property
    def history_len(self):
        return self._history_len

    @property
    def observation_space(self) -> spaces.space.Space:
        return self._observation_space

    @property
    def state_space(self) -> spaces.space.Space:
        return self._state_space

    @property
    def action_space(self) -> spaces.space.Space:
        return self._action_space

    def _prepare_env(self):
        del self.env

        env = L2M2019Env(**self._environment_params, visualize=self._visualize)
        env = wrap_l2m_env(env, **self._wrapper_params)

        self.env = env

    def _prepare_spaces(self):
        self._observation_space = self.env.observation_space
        self._action_space = self.env.action_space

        self._state_space = extend_space(
            self._observation_space, self._history_len
        )

    def _process_action(self, action):
        action = action.copy()

        if self.action_mean is not None \
                and self.action_std is not None:
            action = action * (self.action_std + 1e-8) + self.action_mean

        return action

    def reset(self):
        if self._episode % self._reload_period == 0:
            self._prepare_env()

            if self._difficulty_probabilities is not None:
                difficulty = np.random.choice(
                    [0, 1, 2, 3],
                    p=self._difficulty_probabilities
                )
                self.env.change_model(difficulty=difficulty)

        self._episode += 1
        if self._episode < self._hard_seed_initial_episode:
            self._hard_seeds = set()

        if self._mode == "train" and len(self._hard_seeds) > 0:
            if random.random() < self._hard_seed_probability:
                seed = random.sample(self._hard_seeds, 1)[0]
            else:
                seed = random.randrange(BIG_NUM)
        else:
            seed = random.randrange(BIG_NUM)

        is_ok, is_not_ok_counter = False, 0
        while not is_ok:
            try:
                observation = self.env.reset(seed=seed)
                is_ok = True
            except:
                is_not_ok_counter += 1
                print("something go wrong ", is_not_ok_counter)

        self._episode_step = 0
        self._episode_reward = 0
        self._episode_seed = seed

        return observation

    def step(self, action):
        reward, raw_reward = 0, 0
        action = self._process_action(action)

        for i in range(self._frame_skip):
            observation, r, done, info = self.env.step(action)
            if self._visualize:
                self.env.render()
            reward += r
            raw_reward += info.get("raw_reward", r)

            self._episode_step += 1
            self._episode_reward += info.get("raw_reward", r)

            if done:
                if self._episode_step < self._min_episode_steps \
                        or self._episode_reward < self._min_episode_raw_reward:
                    self._hard_seeds.add(self._episode_seed)
                else:
                    self._hard_seeds.discard(self._episode_seed)
                break

        info["raw_reward"] = raw_reward
        reward += self._step_reward * self._frame_skip
        reward *= self._reward_scale

        return observation, reward, done, info
