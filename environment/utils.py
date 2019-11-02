import numpy as np


INIT_POSE = np.array([
    1.699999999999999956e+00,  # forward speed
    .5,  # rightward speed
    9.023245653983965608e-01,  # pelvis height
    2.012303881285582852e-01,  # trunk lean           0.02
    0 * np.pi / 180,  # [right] hip adduct   0.0
    -6.952390849304798115e-01,  # hip flex             0.6
    -3.231075259785813891e-01,  # knee extend          0.3
    1.709011708233401095e-01,  # ankle flex           0.17
    0 * np.pi / 180,  # [left] hip adduct    0.0
    -5.282323914341899296e-02,  # hip flex             0.05
    -8.041966456860847323e-01,  # knee extend          0.8
    -1.745329251994329478e-01,  # ankle flex           0.1
])


def _obs2reflexobs(obs_dict):
    # refer to LocoCtrl.s_b_keys and LocoCtrl.s_l_keys
    # coordinate in body frame
    #   [0] x: forward
    #   [1] y: leftward
    #   [2] z: upward

    sensor_data = {"body":{}, "r_leg":{}, "l_leg":{}}
    sensor_data["body"]["theta"] = [
        obs_dict["pelvis"]["roll"], # around local x axis
        obs_dict["pelvis"]["pitch"]] # around local y axis

    sensor_data["body"]["d_pos"] = [
        obs_dict["pelvis"]["vel"][0], # local x (+) forward
        obs_dict["pelvis"]["vel"][1]] # local y (+) leftward

    sensor_data["body"]["dtheta"] = [
        obs_dict["pelvis"]["vel"][4], # around local x axis
        obs_dict["pelvis"]["vel"][3]] # around local y axis

    sensor_data["r_leg"]["load_ipsi"] = \
        obs_dict["r_leg"]["ground_reaction_forces"][2]
    sensor_data["l_leg"]["load_ipsi"] = \
        obs_dict["l_leg"]["ground_reaction_forces"][2]

    for s_leg, s_legc in zip(["r_leg", "l_leg"], ["l_leg", "r_leg"]):

        sensor_data[s_leg]["contact_ipsi"] = \
            1 if sensor_data[s_leg]["load_ipsi"] > 0.1 else 0
        sensor_data[s_leg]["contact_contra"] = \
            1 if sensor_data[s_legc]["load_ipsi"] > 0.1 else 0
        sensor_data[s_leg]["load_contra"] = sensor_data[s_legc]["load_ipsi"]

        sensor_data[s_leg]["phi_hip"] = \
            obs_dict[s_leg]["joint"]["hip"] + np.pi
        sensor_data[s_leg]["phi_knee"] = \
            obs_dict[s_leg]["joint"]["knee"] + np.pi
        sensor_data[s_leg]["phi_ankle"] = \
            obs_dict[s_leg]["joint"]["ankle"] + .5*np.pi
        sensor_data[s_leg]["dphi_knee"] = \
            obs_dict[s_leg]["d_joint"]["knee"]

        # alpha = hip - 0.5*knee
        sensor_data[s_leg]["alpha"] = \
            sensor_data[s_leg]["phi_hip"] - .5*sensor_data[s_leg]["phi_knee"]
        dphi_hip = obs_dict[s_leg]["d_joint"]["hip"]
        sensor_data[s_leg]["dalpha"] = \
            dphi_hip - .5*sensor_data[s_leg]["dphi_knee"]
        sensor_data[s_leg]["alpha_f"] = \
            -obs_dict[s_leg]["d_joint"]["hip_abd"] + .5*np.pi

        sensor_data[s_leg]["F_RF"] = obs_dict[s_leg]["RF"]["f"]
        sensor_data[s_leg]["F_VAS"] = obs_dict[s_leg]["VAS"]["f"]
        sensor_data[s_leg]["F_GAS"] = obs_dict[s_leg]["GAS"]["f"]
        sensor_data[s_leg]["F_SOL"] = obs_dict[s_leg]["SOL"]["f"]

    return sensor_data


def _get_all_features(obs_dict, LENGTH0=1.0):
    features = []
    features.append(obs_dict["pelvis"]["height"])  # 0
    features.append(obs_dict["pelvis"]["pitch"])  # 1
    features.append(obs_dict["pelvis"]["roll"])  # 2
    features.append(obs_dict["pelvis"]["vel"][0] / LENGTH0)  # 3
    features.append(obs_dict["pelvis"]["vel"][1] / LENGTH0)  # 4
    features.append(obs_dict["pelvis"]["vel"][2] / LENGTH0)  # 5
    features.append(obs_dict["pelvis"]["vel"][3])  # 6
    features.append(obs_dict["pelvis"]["vel"][4])  # 7
    features.append(obs_dict["pelvis"]["vel"][5])  # 8

    for leg in ["r_leg", "l_leg"]:
        features += obs_dict[leg]["ground_reaction_forces"]  # 9, 10, 11
        features.append(obs_dict[leg]["joint"]["hip_abd"])  # 12
        features.append(obs_dict[leg]["joint"]["hip"])  # 13
        features.append(obs_dict[leg]["joint"]["knee"])  # 14
        features.append(obs_dict[leg]["joint"]["ankle"])  # 15
        features.append(obs_dict[leg]["d_joint"]["hip_abd"])  # 16
        features.append(obs_dict[leg]["d_joint"]["hip"])  # 17
        features.append(obs_dict[leg]["d_joint"]["knee"])  # 18
        features.append(obs_dict[leg]["d_joint"]["ankle"])  # 19
        for MUS in [
            "HAB", "HAD", "HFL", "GLU", "HAM", "RF", "VAS", "BFSH", "GAS",
            "SOL", "TA"
        ]:
            features.append(obs_dict[leg][MUS]["f"])  # 20
            features.append(obs_dict[leg][MUS]["l"])  # 21
            features.append(obs_dict[leg][MUS]["v"])  # 22

    # 97 features
    features = np.array(features)
    return features


def _get_reflex_features(obs_dict):
    sensor_data = _obs2reflexobs(obs_dict)

    features = []
    features.append(sensor_data["body"]["theta"][0])  # 0
    features.append(sensor_data["body"]["theta"][1])  # 1
    features.append(sensor_data["body"]["d_pos"][0])  # 2
    features.append(sensor_data["body"]["d_pos"][1])  # 3
    features.append(sensor_data["body"]["dtheta"][0])  # 4
    features.append(sensor_data["body"]["dtheta"][1])  # 5

    for leg in ["r_leg", "l_leg"]:
        for key in [
            "load_ipsi", "contact_ipsi", "contact_contra",
            "load_contra", "phi_hip", "phi_knee",
            "phi_ankle", "dphi_knee", "alpha",
            "dalpha", "alpha_f", "F_RF",
            "F_VAS", "F_GAS", "F_SOL"
        ]:
            features.append(sensor_data[leg][key])

    # 36 features
    features = np.array(features)
    return features


def _get_osim_features(state_desc):
    res = []

    force_mult = 5e-4
    acc_mult = 1e-2

    # body parts
    body_parts = [
        "head", "torso", "pelvis",
        "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
        "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l",
    ]
    # calculate linear coordinates of pelvis to switch
    # to the reference frame attached to the pelvis
    x, y, z = state_desc["body_pos"]["pelvis"]
    rx, ry, rz = state_desc["body_pos_rot"]["pelvis"]
    res += [y, z]
    res += [rx, ry, rz]
    # 30 components -- relative linear coordinates of body parts (except pelvis)
    for body_part in body_parts:
        if body_part != "pelvis":
            x_, y_, z_ = state_desc["body_pos"][body_part]
            res += [x_ - x, y_ - y, z_ - z]
    # 2 components -- relative linear coordinates of center mass
    x_, y_, z_ = state_desc["misc"]["mass_center_pos"]
    res += [x_ - x, y_ - y, z_ - z]
    # 35 components -- linear velocities of body parts (and center mass)
    for body_part in body_parts:
        res += state_desc["body_vel"][body_part]
    res += state_desc["misc"]["mass_center_vel"]
    # 35 components -- linear accelerations of body parts (and center mass)
    for body_part in body_parts:
        res += state_desc["body_acc"][body_part]
    res += state_desc["misc"]["mass_center_acc"]
    # calculate angular coordinates of pelvis to switch
    # to the reference frame attached to the pelvis
    # 30 components - relative angular coordinates of body parts (except pelvis)
    for body_part in body_parts:
        if body_part != "pelvis":
            rx_, ry_, rz_ = state_desc["body_pos_rot"][body_part]
            res += [rx_ - rx, ry_ - ry, rz_ - rz]
    # 33 components -- linear velocities of body parts
    for body_part in body_parts:
        res += state_desc["body_vel_rot"][body_part]
    # 33 components -- linear accelerations of body parts
    for body_part in body_parts:
        res += state_desc["body_acc_rot"][body_part]

    # joints
    for joint_val in ["joint_pos", "joint_vel", "joint_acc"]:
        for joint in [
            "ground_pelvis",
            "hip_r", "knee_r", "ankle_r",
            "hip_l", "knee_l", "ankle_l"
        ]:
            res += state_desc[joint_val][joint][:3]

    # muscles
    muscles = [
        "abd_r", "add_r", "hamstrings_r",
        "bifemsh_r", "glut_max_r", "iliopsoas_r",
        "rect_fem_r", "vasti_r", "gastroc_r",
        "soleus_r", "tib_ant_r",
        "abd_l", "add_l", "hamstrings_l",
        "bifemsh_l", "glut_max_l", "iliopsoas_l",
        "rect_fem_l", "vasti_l", "gastroc_l",
        "soleus_l", "tib_ant_l",
    ]
    for muscle in muscles:
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]
    for muscle in muscles:
        res += [state_desc["muscles"][muscle]["fiber_force"] * force_mult]

    forces = [
        "abd_r", "add_r", "hamstrings_r",
        "bifemsh_r", "glut_max_r", "iliopsoas_r",
        "rect_fem_r", "vasti_r", "gastroc_r",
        "soleus_r", "tib_ant_r", "foot_r",
        "abd_l", "add_l", "hamstrings_l",
        "bifemsh_l", "glut_max_l", "iliopsoas_l",
        "rect_fem_l", "vasti_l", "gastroc_l",
        "soleus_l", "tib_ant_l", "foot_l",
        "HipLimit_r", "HipLimit_l",
        "KneeLimit_r", "KneeLimit_l",
        "AnkleLimit_r", "AnkleLimit_l",
        "HipAddLimit_r", "HipAddLimit_l"
    ]

    # forces
    for force in forces:
        f = state_desc["forces"][force]
        if len(f) == 1:
            res += [f[0] * force_mult]

    #     res = (np.array(res) - obs_means) / obs_stds
    res = np.array(res)
    # 391 features
    return res


def _prepare_observation(
    observation_dict,
    *,
    LENGTH0=1.0,
    vector_field_mode=None,
    crop_size=0,
    features_mode=None
):
    # Augmented environment from the L2R challenge
    vector_field = np.array(observation_dict["v_tgt_field"]).reshape(2, 11, 11)
    if vector_field_mode is not None:
        if vector_field_mode == "forward_crop":
            vector_field = vector_field[:, 5:, 5 - crop_size:5 + crop_size + 1]
        elif vector_field_mode == "central_crop":
            vector_field = \
                vector_field[
                :,
                5 - crop_size:5 + crop_size + 1,
                5 - crop_size:5 + crop_size + 1
                ]
        else:
            raise ValueError(
                "Wrong mode supplied: {}".format(vector_field_mode))
        # features = np.concatenate([features, vector_field.ravel()])

    if features_mode is None or features_mode == "all":
        features = _get_all_features(observation_dict, LENGTH0)
    elif features_mode == "reflex":
        features = _get_reflex_features(observation_dict)
    elif features_mode == "osim":
        features = None
    else:
        raise ValueError(
            "Wrong mode supplied: {}".format(features_mode))

    return vector_field, features


INIT_DICT = np.load("./assets/init_pose/default.npy", allow_pickle=True)[()]


def _generateY(scale=1):
    # pos taken from the submit server
    init_dict = INIT_DICT.copy()

    # does it matter?
    init_dict["back"] = -0.0872665

    # ranges are taken from
    # https://github.com/stanfordnmbl/osim-rl/blob/master/osim/models/gait14dof22musc_20170320.osim

    lopt = {
        "r_leg": {
            "HAB": 0.0845,
            "HAD": 0.087,
            "HFL": 0.117,
            "GLU": 0.157,
            "HAM": 0.069,
            "RF": 0.076,
            "VAS": 0.099,
            "BFSH": 0.11,
            "GAS": 0.051,
            "SOL": 0.044,
            "TA": 0.068
        },
        "l_leg": {
            "HAB": 0.0845,
            "HAD": 0.087,
            "HFL": 0.117,
            "GLU": 0.157,
            "HAM": 0.069,
            "RF": 0.076,
            "VAS": 0.099,
            "BFSH": 0.11,
            "GAS": 0.051,
            "SOL": 0.044,
            "TA": 0.068
        }
    }

    # len(Y) = 78

    y = np.zeros(78)

    MAX_ANGLE_DIFF = np.pi / 15 * scale
    MAX_ANGLE_VELOCITY_DIFF = np.pi / 20 * scale
    MAX_LINEAR_VELOCITY_DIFF = 0.05 * scale
    MAX_ACTIVATION_DIFF = 0.05 * scale
    MAX_LENGTH_DIFF = 0.05 * scale
    MAX_SHIFT_X = 5 * scale
    MAX_SHIFT_Z = 3 * scale
    # typical muscle length ~ 0.9

    # roughly 0.94
    y[4] = init_dict["pelvis"]["height"]
    y[4] += np.random.uniform(-0.05, 0.05)

    # r ranges are -1.57079633 1.57079633
    y[0] = -init_dict["pelvis"]["pitch"]
    y[1] = -init_dict["pelvis"]["roll"]
    y[0] += np.random.uniform(-MAX_ANGLE_DIFF, MAX_ANGLE_DIFF)
    y[1] += np.random.uniform(-MAX_ANGLE_DIFF, MAX_ANGLE_DIFF)

    y[20] = init_dict["pelvis"]["vel"][0]
    y[22] = -init_dict["pelvis"]["vel"][1]
    y[21] = init_dict["pelvis"]["vel"][2]
    for i in [20, 21, 22]:
        y[i] += np.random.uniform(
            -MAX_ANGLE_VELOCITY_DIFF,
            MAX_ANGLE_VELOCITY_DIFF
        )

    y[17] = -init_dict["pelvis"]["vel"][3]
    y[18] = init_dict["pelvis"]["vel"][4]
    y[19] = init_dict["pelvis"]["vel"][5]
    for i in [17, 18, 19]:
        y[i] += np.random.uniform(
            -MAX_LINEAR_VELOCITY_DIFF,
            MAX_LINEAR_VELOCITY_DIFF
        )

    # range is -2.0943951 2.0943951
    y[6] = -init_dict["r_leg"]["joint"]["hip"]
    y[9] = -init_dict["l_leg"]["joint"]["hip"]

    # range is -2.0943951 2.0943951
    y[7] = -init_dict["r_leg"]["joint"]["hip_abd"]
    y[10] = -init_dict["l_leg"]["joint"]["hip_abd"]

    # range is -2.0943951 0.17453293
    y[13] = init_dict["r_leg"]["joint"]["knee"]
    y[14] = init_dict["l_leg"]["joint"]["knee"]

    # range is -1.57079633 1.57079633
    y[15] = -init_dict["r_leg"]["joint"]["ankle"]
    y[16] = -init_dict["l_leg"]["joint"]["ankle"]

    for i in [7, 10, 13, 14, 15, 16]:
        y[i] += np.random.uniform(-MAX_ANGLE_DIFF, MAX_ANGLE_DIFF)

    y[30] = init_dict["r_leg"]["d_joint"]["knee"]
    y[31] = init_dict["l_leg"]["d_joint"]["knee"]

    y[32] = -init_dict["r_leg"]["d_joint"]["ankle"]
    y[33] = -init_dict["l_leg"]["d_joint"]["ankle"]

    y[26] = -init_dict["l_leg"]["d_joint"]["hip"]
    y[23] = -init_dict["r_leg"]["d_joint"]["hip"]

    y[27] = -init_dict["l_leg"]["d_joint"]["hip_abd"]
    y[24] = -init_dict["r_leg"]["d_joint"]["hip_abd"]

    for i in [23, 24, 26, 27, 30, 31, 32, 33]:
        y[i] += np.random.uniform(
            -MAX_ANGLE_VELOCITY_DIFF,
            MAX_ANGLE_VELOCITY_DIFF
        )

    y[12] = init_dict["back"]

    muscles = [
        "HAB", "HAD", "HAM", "BFSH", "GLU",
        "HFL", "RF", "VAS", "GAS", "SOL", "TA"
    ]

    idx = 35 + 2 * np.arange(len(muscles))
    for (mus, i) in zip(muscles, idx):
        d = np.random.uniform(-MAX_LENGTH_DIFF, MAX_LENGTH_DIFF)
        y[i] = (init_dict["r_leg"][mus]["l"] + d) * lopt["r_leg"][mus]

    idx = 57 + 2 * np.arange(len(muscles))
    for (mus, i) in zip(muscles, idx):
        d = np.random.uniform(-MAX_LENGTH_DIFF, MAX_LENGTH_DIFF)
        y[i] = (init_dict["l_leg"][mus]["l"] + d) * lopt["l_leg"][mus]

    idx_0 = [2, 3, 5, 8, 11, 25, 28, 29]
    idx_005 = [
        34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66,
        68, 70, 72, 74, 76
    ]

    # muscle activations
    for idx in idx_005:
        y[idx] = 0.05 + np.random.uniform(0, MAX_ACTIVATION_DIFF)

    # not used?
    for idx in idx_0:
        y[idx] = 0

    # initial coordinates
    # ranges are -5, 5 for x
    # -3, 3 for z
    y[3] = np.random.uniform(-MAX_SHIFT_X, MAX_SHIFT_X)
    y[5] = np.random.uniform(-MAX_SHIFT_Z, MAX_SHIFT_Z)

    return y
