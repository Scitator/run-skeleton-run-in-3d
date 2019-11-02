from .env import L2MWrapper
from .wrappers import FlattenWrapper


def wrap_l2m_env(
    env,

    # initial position
    random_start_probability=-1,
    random_pose_probability=-1,
    random_pose_scale=0,
    # features space
    use_time_feature=True,
    vector_field_normalization=False,
    vector_field_mode=None,
    crop_size=0,
    features_normalization=False,
    features_mode=None,
    use_extra_vector_field_features=False,
    # reward shaping
    step_reward: float = 0.0,
    pelvis_height_penalty: float = 0.0,
    crossing_legs_penalty: float = 0.0,
    bending_knees_bonus: float = 0.0,
    left_knee_bonus: float = 0.,
    right_knee_bonus: float = 0.,
    bonus_for_knee_angles_scale: float = 0.,
    bonus_for_knee_angles_angle: float = 0.,
    activations_penalty: float = 0.,
    # gym wrappers
    use_flatten_wrapper: bool = False,
    return_flatten_dict: bool = True,
):
    env = L2MWrapper(
        env=env,
        vector_field_normalization=vector_field_normalization,
        features_normalization=features_normalization,
        use_time_feature=use_time_feature,
        random_start_probability=random_start_probability,
        random_pose_probability=random_pose_probability,
        random_pose_scale=random_pose_scale,
        vector_field_mode=vector_field_mode,
        crop_size=crop_size,
        features_mode=features_mode,
        step_reward=step_reward,
        pelvis_height_penalty=pelvis_height_penalty,
        crossing_legs_penalty=crossing_legs_penalty,
        bending_knees_bonus=bending_knees_bonus,
        left_knee_bonus=left_knee_bonus,
        right_knee_bonus=right_knee_bonus,
        bonus_for_knee_angles_scale=bonus_for_knee_angles_scale,
        bonus_for_knee_angles_angle=bonus_for_knee_angles_angle,
        activations_penalty=activations_penalty,
    )

    if use_flatten_wrapper:
        env = FlattenWrapper(env, return_dict=return_flatten_dict)

    return env
