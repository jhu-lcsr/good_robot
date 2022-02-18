import torch
import torch.nn as nn
from learning.models.semantic_map.map_affine import MapAffine
from visualization import Presenter


class MapTransformerBase(nn.Module):

    # TODO: Refactor this entire getting/setting idea
    def __init__(self, source_map_size, world_size_px, world_size_m, dest_map_size=None):
        super(MapTransformerBase, self).__init__()

        if dest_map_size is None:
            dest_map_size = source_map_size

        self.latest_maps = None
        self.latest_map_poses = None

        self.map_affine = MapAffine(
            source_map_size=source_map_size,
            dest_map_size = dest_map_size,
            world_size_px=world_size_px,
            world_size_m = world_size_m)

    def init_weights(self):
        pass

    def reset(self):
        self.latest_maps = None
        self.latest_map_poses = None

    def get_map(self, cam_pose=None, show=""):
        """
        Return the latest map that's been accumulated.
        :param cam_pose: The map will be oriented in the frame of reference of cam_pose before returning
        :return:
        """
        """
        if not self.latest_map_pose == cam_pose:
            map_in_current_frame = self.map_affine(self.latest_map, self.latest_map_pose, cam_pose)
            if show != "":
                Presenter().show_image(map_in_current_frame.data[0, 0:3], show, torch=True, scale=8, waitkey=20)
            return map_in_current_frame, cam_pose
        else:
            return self.latest_map, self.latest_map_pose
        """
        maps, poses = self.get_maps(cam_pose)
        # TODO: Check that this is correct and perhaps deprecate it
        return maps[maps.size(0)-1:maps.size(0)], poses[-1]

    def get_maps(self, cam_poses):
        """
        Return the latest sequence of maps that's been stored.
        :param cam_poses: Each map in the batch will be oriented in the frame of reference of cam_pose_i before returning
        :return:
        """
        #maps = []
        ## TODO: Add proper batch support to map_affine
        #for i, cam_pose in enumerate(cam_poses):
        #    if cam_pose == self.latest_map_poses[i]:
        #        maps.append(self.latest_maps[i])
        #    else:
        #        map_i_in_pose_i = self.map_affine(self.latest_maps[i:i+1], self.latest_map_poses[i:i+1], cam_pose)
        #        maps.append(map_i_in_pose_i)

        maps = self.map_affine(self.latest_maps, self.latest_map_poses, cam_poses)
        return maps, cam_poses

    def set_map(self, map, pose):
        self.latest_maps = map
        self.latest_map_poses = pose

    def set_maps(self, maps, poses):
        self.latest_maps = maps
        self.latest_map_poses = poses

    def add_maps(self, maps, poses):
        self.latest_maps = torch.cat([self.latest_maps, maps], dim=0)
        self.latest_map_poses = self.latest_map_poses + poses

    def forward(self, maps, map_poses, new_map_poses):
        maps = self.map_affine(maps, map_poses, new_map_poses)
        return maps, new_map_poses