from typing import Union
from functools import partial
import random
import math
import time

import numpy as np
from gym import spaces
import gym

from osim.env import L2M2019Env
from .utils import _get_mean_std, _prepare_observation, \
    _get_osim_features, _generateY


VECTOR_BONUS = 9     # = 3^2
DIRECTION_BONUS = 4  # = 2^2
VELOCITY_BONUS = 9   # = 3^2
POSITION_BONUS = 8   #


class L2M2(L2M2019Env):
    reward_shaping_weights = {
        "effort": None,
        "v_tgt_vector": None,
        "v_tgt_direction": None,
        "v_tgt_velocity": None,
        "v_tgt_position": None,
        "v_tgt_position_delta": None,
        "new_v_tgt_field_bonus": None,
        "min_sink_distance": None,
        "max_sink_distance": None,
        "min_time_near_sink": None,
        "max_time_near_sink": None,
        "n_new_target": None,
        "new_target_end": None,
        "speed_threshold": None,
        "distance_threshold": 1.0,
        "step_reward": 1.0,
    }

    def __init__(
        self,
        reward_shaping_weights=None,
        timestep_limit=int(1e4),
        **kwargs
    ):
        if reward_shaping_weights is not None:
            for key, value in reward_shaping_weights.items():
                self.reward_shaping_weights[key] = value
        self.timestep_limit_ = timestep_limit
        self.i_target = 0
        self.previous_t_target = 0
        super().__init__(**kwargs)
        self.vtgt.ver["ver03"]["rng_p_sink_r_th"] = np.array(
            [
                [
                    self.reward_shaping_weights["min_sink_distance"],
                    self.reward_shaping_weights["max_sink_distance"],
                ],
             [-180*np.pi/180, 180*np.pi/180]
            ]
        )
        self.vtgt.ver["ver03"]["n_new_target"] = \
            self.reward_shaping_weights["n_new_target"]
        self.last_distance2target = None
        self.min_distance2target = None
        self._targets_timestamps = []

    @property
    def targets_timestamps(self):
        return self._targets_timestamps

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.last_distance2target = \
            np.linalg.norm(self.vtgt.p_sink - self.vtgt.pose_agent[0:2])
        self.min_distance2target = self.last_distance2target
        self.vtgt.t0_target = np.random.uniform(
            self.reward_shaping_weights["min_time_near_sink"],
            self.reward_shaping_weights["max_time_near_sink"]
        )
        time.sleep(10)
        return obs

    def _reset_y(
        self,
        init_y,
        project=True,
        seed=None,
        obs_as_dict=True,
    ):
        self.t = 0
        self.init_reward()
        self.vtgt.reset(version=self.difficulty, seed=seed)

        self.footstep["n"] = 0
        self.footstep["new"] = False
        self.footstep["r_contact"] = 1
        self.footstep["l_contact"] = 1

        # initialize state
        self.osim_model.state = self.osim_model.model.initializeState()
        state = self.osim_model.get_state()

        Y = state.getY()
        for i in range(state.getNY()):
            Y[i] = init_y[i]

        state.setY(Y)

        self.osim_model.set_state(state)
        self.osim_model.model.equilibrateMuscles(self.osim_model.state)

        self.osim_model.state.setTime(0)
        self.osim_model.istep = 0

        self.osim_model.reset_manager()

        d = super(L2M2019Env, self).get_state_desc()
        pose = np.array([
            d["body_pos"]["pelvis"][0],
            -d["body_pos"]["pelvis"][2],
            d["joint_pos"]["ground_pelvis"][2]
        ])
        self.v_tgt_field, self.flag_new_v_tgt_field = self.vtgt.update(pose)

        self.last_distance2target = \
            np.linalg.norm(self.vtgt.p_sink - self.vtgt.pose_agent[0:2])
        self.min_distance2target = self.last_distance2target
        self.vtgt.t0_target = np.random.uniform(
            self.reward_shaping_weights["min_time_near_sink"],
            self.reward_shaping_weights["max_time_near_sink"]
        )

        time.sleep(10)
        if not project:
            return self.get_state_desc()
        if obs_as_dict:
            return self.get_observation_dict()
        return self.get_observation()

    def set_difficulty(self, difficulty):
        super().set_difficulty(difficulty)
        self.spec.timestep_limit = self.timestep_limit_

    def get_reward_2(self): # for L2M2019 Round 2
        state_desc = self.get_state_desc()
        if not self.get_prev_state_desc():
            return 0

        reward, dense_reward = 0, 0
        dt = self.osim_model.stepsize

        # alive reward
        # should be large enough to search
        # for "success" solutions (alive to the end) first
        # @TODO: MODIFICATION 0
        reward += self.d_reward["alive"]
        dense_reward -= self.d_reward["alive"]  # 0.1s
        # @TODO: MODIFICATION 0

        # effort ~ muscle fatigue ~ (muscle activation)^2
        ACT2 = 0
        for muscle in sorted(state_desc["muscles"].keys()):
            ACT2 += np.square(state_desc["muscles"][muscle]["activation"])
        self.d_reward["effort"] += ACT2 * dt
        self.d_reward["footstep"]["effort"] += ACT2 * dt

        self.d_reward["footstep"]["del_t"] += dt

        # @TODO: MODIFICATION 1
        dense_reward -= ACT2 * self.reward_shaping_weights["effort"]
        # @TODO: MODIFICATION 1

        # reward from velocity (penalize from deviating from v_tgt)
        body_position = [
            state_desc["body_pos"]["pelvis"][0],
            -state_desc["body_pos"]["pelvis"][2]
        ]
        body_velocity_vector = [
            state_desc["body_vel"]["pelvis"][0],
            -state_desc["body_vel"]["pelvis"][2]
        ]
        field_velocity_vector = self.vtgt.get_vtgt(body_position).T[0]

        self.d_reward["footstep"]["del_v"] += \
            (body_velocity_vector - field_velocity_vector) * dt

        # @TODO: MODIFICATION 2
        vector_diff = \
            np.square(body_velocity_vector - field_velocity_vector).sum()
        vector_bonus = VECTOR_BONUS - np.clip(vector_diff, 0, VECTOR_BONUS)

        body_velocity_len = np.linalg.norm(body_velocity_vector)
        field_velocity_len = np.linalg.norm(field_velocity_vector)
        body_velocity_vector /= body_velocity_len
        field_velocity_vector /= field_velocity_len
        # @TODO: we need life BONUS -> 4 reward, do not change
        # 4 - 0 = +4 - same direction
        # 4 - 4 = 0 - diff direction

        if self.reward_shaping_weights["speed_threshold"] is not None:
            field_velocity_len = self.reward_shaping_weights["speed_threshold"]

        direction_bonus = (
            DIRECTION_BONUS
            - np.square(body_velocity_vector - field_velocity_vector).sum()
        )
        velocity_diff = np.square(body_velocity_len - field_velocity_len)
        velocity_bonus = \
            VELOCITY_BONUS - np.clip(velocity_diff, 0, VELOCITY_BONUS)

        distance2target = \
            np.linalg.norm(self.vtgt.p_sink - self.vtgt.pose_agent[0:2])
        # delta > 0 - good
        # delta < 0 - bad
        position_delta = self.last_distance2target - distance2target
        position_bonus = POSITION_BONUS - min(distance2target, POSITION_BONUS)
        position_bonus = np.square(position_bonus)

        # If we achieved new sink, hack its t0 required
        if self.i_target < self.vtgt.i_target:
            self.vtgt.t0_target = np.random.uniform(
                self.reward_shaping_weights["min_time_near_sink"],
                self.reward_shaping_weights["max_time_near_sink"]
            )

        if distance2target < self.reward_shaping_weights["distance_threshold"]:
            direction_bonus = DIRECTION_BONUS              # max direction bonus
            velocity_bonus = VELOCITY_BONUS    # max velocity bonus
            # position_bonus = POSITION_BONUS              # max position bonus

        self.min_distance2target = \
            min(self.last_distance2target, distance2target)
        self.last_distance2target = distance2target
        self.i_target = self.vtgt.i_target
        self.previous_t_target = self.vtgt.t_target

        vector_bonus /= VECTOR_BONUS
        direction_bonus /= DIRECTION_BONUS
        velocity_bonus /= VELOCITY_BONUS
        position_bonus /= np.square(POSITION_BONUS)

        dense_reward += \
            vector_bonus * self.reward_shaping_weights["v_tgt_vector"]
        dense_reward += \
            direction_bonus * self.reward_shaping_weights["v_tgt_direction"]
        dense_reward += \
            velocity_bonus * self.reward_shaping_weights["v_tgt_velocity"]
        dense_reward += \
            position_bonus * self.reward_shaping_weights["v_tgt_position"]
        dense_reward += \
            position_delta * self.reward_shaping_weights["v_tgt_position_delta"]
        # @TODO: MODIFICATION 2

        # simulation ends successfully
        flag_success = (
            # model did not fall down
            not self.is_done()
            # reached end of simulation
            and (self.osim_model.istep >= self.spec.timestep_limit)
            # took more than 5 footsteps (to prevent standing still)
            and self.footstep["n"] > 5
        )

        step_reward = 0
        # footstep reward (when made a new step)
        if self.footstep["new"] or flag_success:
            # footstep reward: so that solution does not avoid making footsteps
            # scaled by del_t,
            # so that solution does not get higher rewards
            # by making unnecessary (small) steps
            reward_footstep_0 = (
                self.d_reward["weight"]["footstep"]
                * self.d_reward["footstep"]["del_t"]
            )

            # deviation from target velocity
            # the average velocity a step
            # (instead of instantaneous velocity) is used
            # as velocity fluctuates within a step in normal human walking
            #reward_footstep_v =
            # -self.reward_w["v_tgt"]*(self.footstep["del_vx"]**2)
            reward_footstep_v = (
                -self.d_reward["weight"]["v_tgt_R2"]
                * np.linalg.norm(self.d_reward["footstep"]["del_v"])
                /self.LENGTH0
            )

            # panalize effort
            reward_footstep_e = (
                -self.d_reward["weight"]["effort"]
                * self.d_reward["footstep"]["effort"]
            )

            self.d_reward["footstep"]["del_t"] = 0
            self.d_reward["footstep"]["del_v"] = 0
            self.d_reward["footstep"]["effort"] = 0

            step_reward = \
                reward_footstep_0 + reward_footstep_v + reward_footstep_e
            reward += step_reward

        # @TODO: MODIFICATION 3
        # task bonus: if stayed enough at the first target
        if self.flag_new_v_tgt_field:
            reward += 500
            dense_reward -= 500
        # @TODO: MODIFICATION 3

        # @TODO: MODIFICATION FINAL
        # dense_reward += reward
        dense_reward += step_reward * self.reward_shaping_weights["step_reward"]
        # @TODO: MODIFICATION FINAL

        return dense_reward, reward

    def step(self, action, project=True, obs_as_dict=True):
        action_mapped = [action[i] for i in self.act2mus]

        _, reward, done, info = super(L2M2019Env, self).step(
            action_mapped, project=project, obs_as_dict=obs_as_dict)
        self.t += self.osim_model.stepsize
        self.update_footstep()

        d = super(L2M2019Env, self).get_state_desc()
        self.pose = np.array([
            d["body_pos"]["pelvis"][0],
            -d["body_pos"]["pelvis"][2],
            d["joint_pos"]["ground_pelvis"][2]
        ])
        self.v_tgt_field, self.flag_new_v_tgt_field = \
            self.vtgt.update(self.pose)

        if project:
            if obs_as_dict:
                obs = self.get_observation_dict()
            else:
                obs = self.get_observation()
        else:
            obs = self.get_state_desc()

        if self.flag_new_v_tgt_field:
            self._targets_timestamps.append(int(self.t * 100))

        # @TODO: MODIFICATION – end of episode
        if self.reward_shaping_weights["new_target_end"]:
            done = done or self.flag_new_v_tgt_field
            if self.flag_new_v_tgt_field:
                dense_reward, reward = reward
                reward += 500
                reward = dense_reward, reward

        return obs, reward, done, info


class L2MWrapper(gym.Wrapper):
    def __init__(
        self,
        env=None,
        vector_field_normalization: Union[bool, str] = False,
        features_normalization: Union[bool, str] = False,
        use_time_feature: bool = True,
        random_start_probability: float = -1,
        random_pose_probability: float = -1,
        random_pose_scale: float = 0,
        vector_field_mode: str = None,
        crop_size: int = 0,
        features_mode: str = None,
        step_reward: float = 0.0,
        pelvis_height_penalty: float = 0.0,
        crossing_legs_penalty: float = 0.0,
        bending_knees_bonus: float = 0.0,
        left_knee_bonus: float = 0.,
        right_knee_bonus: float = 0.,
        bonus_for_knee_angles_scale: float = 0.,
        bonus_for_knee_angles_angle: float = 0.,
        activations_penalty: float = 0.,
    ):
        super().__init__(env)
        if vector_field_mode is None:
            vector_field_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2, 11, 11),
                dtype=env.observation_space.dtype
            )
        else:
            if vector_field_mode == "forward_crop":
                vector_field_space = gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(2, 6, 2 * crop_size + 1),
                    dtype=env.observation_space.dtype
                )
            elif vector_field_mode == "central_crop":
                vector_field_space = gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(2, 2 * crop_size + 1, 2 * crop_size + 1),
                    dtype=env.observation_space.dtype
                )
            else:
                raise ValueError(f"Wrong mode supplied: {vector_field_mode}")

        features_dim = {
            "all": 97,
            "reflex": 36,
            "osim": 391
        }
        self._features_mode = features_mode
        features_size = features_dim[features_mode]
        features_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(features_size + bool(use_time_feature), ),  # env + 1(time)
            dtype=env.observation_space.dtype
        )
        self.observation_space = spaces.Dict(
            {
                "vector_field": vector_field_space,
                "features": features_space
            }
        )
        self._use_time_feature = use_time_feature
        self._random_start_probability = random_start_probability
        self._random_pose_probability = random_pose_probability
        self._random_pose_scale = random_pose_scale
        self._time_step = 0
        self._step_limit = self.env.spec.timestep_limit

        self._prepare_observation = partial(
            _prepare_observation,
            LENGTH0=env.LENGTH0,
            vector_field_mode=vector_field_mode,
            crop_size=crop_size,
            features_mode=features_mode
        )

        vector_field_len = 2 * 11 * 11
        if isinstance(
            vector_field_normalization, bool
        ) and vector_field_normalization:
            vector_field_mean, vector_field_std = _get_mean_std(
                low=self.env.observation_space.low[:vector_field_len],
                high=self.env.observation_space.high[:vector_field_len]
            )
            self.vector_field_mean, self.vector_field_std = \
                vector_field_mean.reshape(vector_field_space.shape), \
                vector_field_std.reshape(vector_field_space.shape)
        elif isinstance(vector_field_normalization, str):
            self.vector_field_mean = \
                np.load(f"{vector_field_normalization}/mean.npy")
            self.vector_field_std = \
                np.load(f"{vector_field_normalization}/std.npy")
        else:
            self.vector_field_mean, self.vector_field_std = None, None
        if isinstance(features_normalization, bool) and features_normalization:
            self.features_mean, self.features_std = _get_mean_std(
                low=self.env.observation_space.low[vector_field_len:],
                high=self.env.observation_space.high[vector_field_len:]
            )
        elif isinstance(features_normalization, str):
            self.features_mean = \
                np.load(f"{features_normalization}/mean.npy")
            self.features_std = \
                np.load(f"{features_normalization}/std.npy")
        else:
            self.features_mean, self.features_std = None, None

        # reward shaping
        self.step_reward = step_reward
        self.pelvis_height_penalty = pelvis_height_penalty
        self.crossing_legs_penalty = crossing_legs_penalty
        self.bending_knees_bonus = bending_knees_bonus
        self.activations_penalty = activations_penalty
        self.left_knee_bonus = left_knee_bonus
        self.right_knee_bonus = right_knee_bonus
        self.bonus_for_knee_angles_scale = bonus_for_knee_angles_scale
        self.knees_angle_bonus = bonus_for_knee_angles_angle

    def _normalize_observation(self, vector_field, features):
        if self.vector_field_mean is not None:
            vector_field = \
                (vector_field - self.vector_field_mean) \
                / (self.vector_field_std + 1e-8)
        if self.features_mean is not None:
            features = \
                (features - self.features_mean) / (self.features_std + 1e-8)
        return vector_field, features

    def _process_observation(
        self,
        *,
        observation_dict=None,
        state_desc=None,
        time_step=None,
    ):
        if self._features_mode == "osim":
            vector_field, _ = \
                self._prepare_observation(state_desc)
            features = _get_osim_features(state_desc)
        else:
            vector_field, features = \
                self._prepare_observation(observation_dict)

        vector_field, features = self._normalize_observation(
            vector_field, features
        )
        if self._use_time_feature:
            features = features.tolist()
            features.append(time_step / self._step_limit * 2 - 1.0)
            features = np.array(features)

        observation = {"vector_field": vector_field, "features": features}
        return observation

    def _process_reward(self, *, reward, state_desc):

        # pelvis height penalty
        pelvis_y = state_desc["body_pos"]["pelvis"][1]
        pelvis_height_penalty = np.minimum(pelvis_y - 0.85, 0.0)
        reward += self.pelvis_height_penalty * pelvis_height_penalty

        # crossing legs penalty
        pelvis_xy = np.array(state_desc["body_pos"]["pelvis"])
        left = np.array(state_desc["body_pos"]["toes_l"]) - pelvis_xy
        right = np.array(state_desc["body_pos"]["toes_r"]) - pelvis_xy
        axis = np.array(state_desc["body_pos"]["head"]) - pelvis_xy

        left /= np.linalg.norm(left)
        right /= np.linalg.norm(right)
        axis /= np.linalg.norm(axis)
        cross = np.cross(left, right)
        cross /= np.linalg.norm(cross)

        cross_legs_penalty = cross.dot(axis)
        cross_legs_penalty = np.minimum(cross_legs_penalty, 0.1)
        cross_legs_penalty += 0.2  # @TODO: trust me, I"m engineer
        reward += self.crossing_legs_penalty * cross_legs_penalty

        # bending knees bonus
        r_knee_flexion = \
            np.minimum(state_desc["joint_pos"]["knee_r"][0], 0.)
        l_knee_flexion = \
            np.minimum(state_desc["joint_pos"]["knee_l"][0], 0.)
        bend_knees_bonus = np.abs(r_knee_flexion + l_knee_flexion)
        reward += self.bending_knees_bonus * bend_knees_bonus

        reward += \
            self.bonus_for_knee_angles_scale \
            * math.exp(-((r_knee_flexion + self.knees_angle_bonus) * 6.0) ** 2)
        reward += \
            self.bonus_for_knee_angles_scale \
            * math.exp(-((l_knee_flexion + self.knees_angle_bonus) * 6.0) ** 2)

        r_knee_flexion = math.fabs(state_desc["joint_vel"]["knee_r"][0])
        l_knee_flexion = math.fabs(state_desc["joint_vel"]["knee_l"][0])
        reward += r_knee_flexion * self.right_knee_bonus
        reward += l_knee_flexion * self.left_knee_bonus

        # we cannot compute this for expert trajectories, so hotfix
        if self.activations_penalty > 0:
            reward -= np.sum(
                np.array(self.env.osim_model.get_activations()) ** 2
            ) * self.activations_penalty

        reward += self.step_reward

        return reward

    def reset(self, **kwargs):
        self._time_step = 0
        init_pose = None

        if random.random() < self._random_start_probability:
            scale = random.random()
            init_y = _generateY(scale=scale)
            observation_dict = self.env._reset_y(**kwargs, init_y=init_y)
        else:
            observation_dict = self.env.reset(**kwargs, init_pose=init_pose)

        state_desc = self.env.get_state_desc()
        self.observation_dict = observation_dict

        observation = self._process_observation(
            observation_dict=observation_dict,
            state_desc=state_desc,
            time_step=self._time_step
        )

        return observation

    def step(self, action, **kwargs):
        self._time_step += 1

        observation_dict, reward, done, info = self.env.step(action, **kwargs)
        self.observation_dict = observation_dict

        dense_reward, reward = reward
        info["raw_reward"] = reward

        state_desc = self.env.get_state_desc()
        observation = self._process_observation(
            observation_dict=observation_dict,
            state_desc=state_desc,
            time_step=self._time_step
        )

        reward = self._process_reward(
            reward=dense_reward,
            state_desc=state_desc)

        return observation, reward, done, info
