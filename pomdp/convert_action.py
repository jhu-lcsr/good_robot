import numpy as np

from parameters.parameter_server import get_current_parameters


def normalize_action(action):
    """
    Given an action executed in AirSim, produce an equivalent action in Neural Network R^3 space
    :param action:
    :return:
    """
    ret = np.zeros(3)
    params = get_current_parameters()["Dynamics"]
    ret[0] = action[0] / params["max_vel_x"]
    ret[2] = action[2] / params["max_vel_theta"]

    ret[0] = np.clip(ret[0], 0, 1)
    ret[2] = np.clip(ret[2], -1, 1)

    return ret


def unnormalize_action(action):
    """
    Given an action in R^3 [0-1] space, produce the correct AirSim action
    :param action:
    :return:
    """
    ret = np.zeros(3)
    params = get_current_parameters()["Dynamics"]
    ret[0] = action[0] * params["max_vel_x"]
    ret[2] = action[2] * params["max_vel_theta"]

    ret[0] = np.clip(ret[0], 0, params["max_vel_x"])
    ret[2] = np.clip(ret[2], -params["max_vel_theta"], params["max_vel_theta"])
    #print("action in: ", action, " action out: ", ret)

    return ret
