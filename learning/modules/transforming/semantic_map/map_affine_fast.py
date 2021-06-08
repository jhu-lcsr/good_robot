import numpy as np
import torch
from torch import nn as nn

from learning.inputs.common import empty_float_tensor, np_to_tensor
from learning.inputs.pose import Pose
from learning.modules.affine_2d import Affine2D
from transformations import get_affine_trans_2d, get_affine_rot_2d, poses_m_to_px

from utils.simple_profiler import SimpleProfiler


PROFILE = False


class MapAffine(nn.Module):

    # TODO: Cleanup unused run_params
    def __init__(self, map_size, world_size_px, world_size_m):
        super(MapAffine, self).__init__()
        self.map_size = map_size
        self.world_size_px = world_size_px
        self.world_size_m = world_size_m

        self.affine_2d = Affine2D()

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

    def pose_2d_to_mat_np(self, pose_2d, inv=False):
        pos = pose_2d.position
        yaw = pose_2d.orientation

        # Transform the img so that the drone's position ends up at the origin
        # TODO: Add batch support
        t1 = get_affine_trans_2d(-pos)

        # Rotate the img so that it's aligned with the drone's orientation
        yaw = -yaw
        t2 = get_affine_rot_2d(-yaw)

        # Translate the img so that it's centered around the drone
        t3 = get_affine_trans_2d([self.map_size/2, self.map_size/2])

        mat = np.dot(t3, np.dot(t2, t1))

        # Swap x and y axes (because of the BxCxHxW a.k.a BxCxYxX convention)
        swapmat = mat[[1,0,2], :]
        mat = swapmat[:, [1,0,2]]

        if inv:
            mat = np.linalg.inv(mat)

        return mat

    def get_old_to_new_pose_mat(self, old_pose, new_pose):
        old_T_inv = self.pose_2d_to_mat_np(old_pose, inv=True)
        new_T = self.pose_2d_to_mat_np(new_pose, inv=False)
        mat = np.dot(new_T, old_T_inv)
        #mat = new_T
        mat_t = np_to_tensor(mat)
        return mat_t

    def get_canonical_frame_pose(self):
        pos = np.asarray([self.map_size/2, self.map_size/2])
        rot = np.asarray([0])

        return Pose(pos, rot)

    def forward(self, maps, map_pose, cam_pose):
        """
        Affine transform the map from being centered around map_pose in the canonocial map frame to
        being centered around cam_pose in the canonical map frame.
        Canonical map frame is the one where the map origin aligns with the environment origin, but the env may
        or may not take up the entire map.
        :param map: map centered around the drone in map_pose
        :param map_pose: the previous drone pose in canonical map frame
        :param cam_pose: the new drone pose in canonical map frame
        :return:
        """

        # TODO: Handle the case where cam_pose is None and return a map in the canonical frame
        self.prof.tick("out")
        batch_size = maps.size(0)
        affine_matrices = torch.zeros([batch_size, 3, 3]).to(maps.device)

        self.prof.tick("init")
        for i in range(batch_size):

            # Convert the pose from airsim coordinates to the image pixel coordinages
            # If the pose is None, use the canonical pose (global frame)
            if map_pose is not None and map_pose[i] is not None:
                map_pose_i = map_pose[i].numpy()
                map_pose_img = poses_m_to_px(map_pose_i, self.map_size, [self.world_size_px, self.world_size_px], self.world_size_m)
            else:
                map_pose_img = self.get_canonical_frame_pose()

            if cam_pose is not None and cam_pose[i] is not None:
                cam_pose_i = cam_pose[i].numpy()
                cam_pose_img = poses_m_to_px(cam_pose_i, self.map_size, [self.world_size_px, self.world_size_px], self.world_size_m)
            else:
                cam_pose_img = self.get_canonical_frame_pose()

            self.prof.tick("pose")

            # Get the affine transformation matrix to transform the map to the new camera pose
            affine_i = self.get_old_to_new_pose_mat(map_pose_img, cam_pose_img)
            affine_matrices[i] = affine_i
            self.prof.tick("affine")

        # TODO: Do the same with OpenCV and compare results for testing

        # Apply the affine transformation on the map
        maps_out = self.affine_2d(maps, affine_matrices)

        self.prof.tick("affine_sample")
        self.prof.loop()
        self.prof.print_stats(20)

        return maps_out