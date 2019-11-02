from functools import reduce

import numpy as np
import gym
from gym import spaces


class ObservationWrapper(gym.Wrapper):
    @property
    def targets_timestamps(self):
        return self.env.targets_timestamps

    def observation(self, observation):
        raise NotImplementedError

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        return self.observation(observation), reward, done, info

    def _process_observation(self, **kwargs):
        observation = self.env._process_observation(**kwargs)
        observation = self.observation(observation)
        return observation

    def _process_reward(self, **kwargs):
        return self.env._process_reward(**kwargs)

    def _get_trajectories(self, **kwargs):
        trajectories = self.env._get_trajectories(**kwargs)
        trajectories_ = {}
        for name, trajectory in trajectories.items():
            observations, actions, rewards, dones = trajectory
            observations = np.array(list(map(
                lambda x: self.observation(x),
                observations
            )))
            trajectory = observations, actions, rewards, dones
            trajectories_[name] = trajectory
        return trajectories_


class FlattenWrapper(ObservationWrapper):
    def __init__(self, env, return_dict: bool = True):
        super().__init__(env)
        self.env = env
        self.return_dict = return_dict

        observation_size = \
            reduce(
                lambda x, y: x * y,
                env.observation_space["features"].shape) \
            + reduce(
                lambda x, y: x*y,
                env.observation_space["vector_field"].shape)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_size, ),
            dtype=env.observation_space["features"].dtype
        )
        if self.return_dict:
            self.observation_space = \
                spaces.Dict({"features": self.observation_space})

    def observation(self, observation):
        observation = np.append(
            observation["vector_field"].flatten(),
            observation["features"].flatten()
        )
        if self.return_dict:
            observation = {"features": observation}
        return observation
