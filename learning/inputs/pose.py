import numpy as np
import torch
from torch.autograd import Variable
from transforms3d import quaternions, euler


class Pose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation

    def __eq__(self, other):
        if other is None:
            return False

        poseq = self.position == other.position
        roteq = self.orientation == other.orientation
        # For numpy arrays, torch ByteTensors and variables containing ByteTensors
        if hasattr(poseq, "all"):
            poseq = poseq.all()
            roteq = roteq.all()
        return poseq and roteq

    def __getitem__(self, i):
        if type(i) in [torch.ByteTensor, torch.cuda.ByteTensor]:
            pos = self.position[i[:, np.newaxis].expand_as(self.position)].view([-1, 3])
            rot = self.orientation[i[:, np.newaxis].expand_as(self.orientation)].view([-1, 4])
            return Pose(pos, rot)
        else:
            return Pose(self.position[i], self.orientation[i])

    def cpu(self):
        self.position = self.position.cpu()
        self.orientation = self.orientation.cpu()
        return self

    def cuda(self, device=None):
        self.position = self.position.cuda(device)
        self.orientation = self.orientation.cuda(device)
        return self

    def to_torch(self):
        position = torch.from_numpy(self.position)
        orientation = torch.from_numpy(self.orientation)
        return Pose(position, orientation)

    def to_var(self):
        position = Variable(self.position)
        orientation = Variable(self.orientation)
        return Pose(position, orientation)

    def repeat_np(self, batch_size):
        position = np.tile(self.position[np.newaxis, :], [batch_size, 1])
        orientation = np.tile(self.orientation[np.newaxis, :], [batch_size, 1])
        return Pose(position, orientation)

    def numpy(self):
        pos = self.position
        rot = self.orientation
        if isinstance(pos, Variable):
            pos = pos.data
            rot = rot.data
        if hasattr(pos, "cuda"):
            pos = pos.cpu().numpy()
            rot = rot.cpu().numpy()
        return Pose(pos, rot)

    def __len__(self):
        if self.position is None:
            return 0
        return len(self.position)

    def __str__(self):
        return "Pose " + str(self.position) + " : " + str(self.orientation)

def get_noisy_poses_np(clean_poses, position_variance, orientation_variance):
    noisy_pos = []
    noisy_rot = []
    for pose in clean_poses:
        pos_eta = np.random.normal([0] * 3, [position_variance] * 3)
        rot_eta = np.random.normal([0] * 3, [0] * 2 + [orientation_variance])

        rpy = np.asarray(euler.quat2euler(pose.orientation))
        rpy_n = rpy + rot_eta
        rot_noisy = euler.euler2quat(rpy_n[0], rpy_n[1], rpy_n[2])
        pos_noisy = pose.position + pos_eta
        noisy_pos.append(pos_noisy)
        noisy_rot.append(rot_noisy)

    noisy_pos = np.stack(noisy_pos).astype(np.float32)
    noisy_rot = np.stack(noisy_rot).astype(np.float32)
    return Pose(noisy_pos, noisy_rot)

def stack_poses_np(poses):
    pos = []
    rot = []
    for pose in poses:
        pos.append(pose.position)
        rot.append(pose.orientation)
    return Pose(np.stack(pos), np.stack(rot))

def get_noisy_poses_torch(clean_poses, position_variance, orientation_variance, cuda=False, cuda_device=None):
    poses_np = clean_poses.numpy()
    noisy_np = get_noisy_poses_np(poses_np, position_variance, orientation_variance)
    noisy_t = noisy_np.to_torch().to_var()
    if cuda:
        noisy_t = noisy_t.cuda(cuda_device)
    return noisy_t


def get_pose_noise_np(numenvs, poses_per_env, pos_variance, rot_variance):
    all_env_poses = []
    for env_id in range(numenvs):
        env_poses = []
        print("env: ", env_id)
        for i in range(poses_per_env):
            pos_eta = np.random.normal([0] * 3, [pos_variance] * 3)
            rot_eta = np.random.normal([0] * 3, [rot_variance] * 3)
            noise = np.concatenate([pos_eta, rot_eta])
            env_poses.append(noise)
        all_env_poses.append(np.asarray(env_poses))
    return all_env_poses