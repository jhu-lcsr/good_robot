from learning.inputs.pose import Pose
import numpy as np
import geometry
from scipy.spatial import distance
from utils.paths import condense_path, get_closest_point_in_path

"""
Given a drone pose and a reference trajectory, generate a smooth trajectory starting from the drone
pose that follows the reference trajectory.
"""


def get_dynamic_ground_truth(path, pose):
    """
    :param path: Nx2 matrix ground truth trajectory
    :param pose: pose.position - 2D vector of X/Y, pose.orientation - yaw angle (radians)
    :return: Mx2 matrix of generated ground truth that tries to follow the path
    """
    counter = 0
    yaw_angle = 2.5
    new_traj = np.array(pose)
    new_traj = np.vstack([new_traj, np.array(pose)])
    distance_matrix = distance.cdist(path, np.array(pose).reshape(1,2))
    if distance_matrix.shape[0] > 1:
        counter = np.argsort(distance_matrix.flatten())[0]

    while counter < path.shape[0] - 1:
        c, s = np.cos(yaw_angle), np.sin(yaw_angle)
        R = np.array(((c, -s), (s, c)))
        current_ref = path[counter]
        next_ref = path[counter+1]
        origin_force = np.dot(R, (next_ref - current_ref))
        shift = next_ref - new_traj[-1]
        combine_force = origin_force + shift
        next_pos = new_traj[-1] + combine_force * \
                   (np.sqrt(origin_force.dot(origin_force)))/np.sqrt(combine_force.dot(combine_force))
        new_traj = np.vstack([new_traj, np.array(next_pos)])
        dynamic_v = (new_traj[-1] - new_traj[-2])/np.linalg.norm((new_traj[-1] - new_traj[-2]))
        truth_v = (next_ref-current_ref)/np.linalg.norm((next_ref-current_ref))
        yaw_angle = np.arccos(np.clip(np.dot(dynamic_v, truth_v), -1.0, 1.0))
        counter += 1
    if np.linalg.norm(path[-1] - new_traj[-1]) > 10:
        next_pos = path[-1]
        new_traj = np.vstack([new_traj, np.array(next_pos)])
    return new_traj[1:]


def get_dynamic_ground_truth_smooth(path, pose):
    yaw_angle = 2.5
    new_traj = np.array(pose)
    new_traj = np.vstack([new_traj, np.array(pose)])
    count = 0
    while np.linalg.norm(path[-1] - new_traj[-1]) > 20 and count < 1000:
        count += 1
        c, s = np.cos(yaw_angle), np.sin(yaw_angle)
        R = np.array(((c, -s), (s, c)))
        distance_matrix = distance.cdist(path, new_traj[-1].reshape(1, 2))
        pointer = np.argsort(distance_matrix.flatten())[0]
        if pointer >= len(path)-1:
            current_ref = path[-2]
            next_ref = path[-1]
        else:
            current_ref = path[pointer]
            next_ref = path[pointer+1]
        origin_force = np.dot(R, (next_ref - current_ref))
        shift = next_ref - new_traj[-1]
        last_pos = new_traj[-1]
        for i in range(5):
            combine_force = origin_force + (shift * np.linalg.norm(shift)/(np.linalg.norm(origin_force) + 1e-9)) * ((i/5) ** 2)
            combine_force = combine_force/(np.linalg.norm(combine_force) + 1e-18)
            next_pos = last_pos + combine_force ** (5-i)
            new_traj = np.vstack([new_traj, np.array(next_pos)])
        dynamic_v = (new_traj[-1] - new_traj[-6])/(np.linalg.norm((new_traj[-1] - new_traj[-6])) + 1e-18)
        truth_v = (next_ref-current_ref)/(np.linalg.norm((next_ref-current_ref)) + 1e-18)
        yaw_angle = np.arccos(np.clip(np.dot(dynamic_v, truth_v), -1.0, 1.0))
    return new_traj[1:]


def get_dynamic_ground_truth_v2(path, position_cf):
    yaw = 0
    start_dir = geometry.yaw_to_vec(yaw)[:2]

    if len(path) == 0:
        return np.asarray([position_cf, position_cf])

    path = condense_path(path)

    current_pos = position_cf
    new_traj = np.array(position_cf)
    new_traj = np.vstack([new_traj, np.array(position_cf)])

    counter = get_closest_point_in_path(path, position_cf)
    current_vel = start_dir * 10
    current_vel = np.zeros(2)

    count = 0
    while np.linalg.norm(path[-1] - new_traj[-1]) > 20 and count < 10000 and counter < len(path) - 1:
        count += 1

        LOOKAHEAD = 5
        if LOOKAHEAD + counter > len(path) - 1:
            LOOKAHEAD = len(path) - 1 - counter

        dir_along_path = path[counter + LOOKAHEAD] - path[counter]
        dir_along_path /= (np.linalg.norm(dir_along_path) + 1e-9)

        dir_towards_path = path[counter + LOOKAHEAD] - current_pos
        dir_towards_path /= (np.linalg.norm(dir_towards_path) + 1e-9)

        lamda = 0.2
        force = lamda * dir_along_path + (1 - lamda) * dir_towards_path

        current_vel += force

        MAX_VEL = 20
        if np.linalg.norm(current_vel) > MAX_VEL:
            current_vel /= np.linalg.norm(current_vel)
            current_vel *= MAX_VEL

        new_pos = current_pos + current_vel * 0.1
        new_traj = np.vstack([new_traj, new_pos])
        current_pos = new_pos

        dst_to_next = np.linalg.norm(path[counter + 1] - current_pos)
        dst_to_this = np.linalg.norm(path[counter] - current_pos)
        if dst_to_next < dst_to_this:
            counter += 1

    return new_traj