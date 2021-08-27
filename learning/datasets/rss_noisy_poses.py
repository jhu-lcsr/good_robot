import os
from data_io.paths import get_noisy_pose_path
import pickle

"""
To run the experiment in RSS 2018 paper where we test resilience to pose noise,
we need to pre-compute and save the noise vectors for every training example.

We can't just add noise during training - that would be regularization, since the
noise would be different at every epoch. That's what these functions are for.
"""

def save_noisy_poses(poses):
    path = get_noisy_pose_path()

    os.makedirs(path, exist_ok=True)

    for i in range(len(poses)):
        filename = "poses_" + str(i) + ".pickle"
        fullfile = os.path.join(path, filename)
        with open(fullfile, "wb") as fp:
            pickle.dump(poses[i], fp)


def load_noisy_poses(env_id):
    path = get_noisy_pose_path()
    filename = "poses_" + str(env_id) + ".pickle"
    fullfile = os.path.join(path, filename)
    with open(fullfile, "rb") as fp:
        env_poses = pickle.load(fp)
    return env_poses