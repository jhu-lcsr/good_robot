import torch
from learning.modules.map_transformer_base import MapTransformerBase

from visualization import Presenter

class IdentityMapAccumulator(MapTransformerBase):
    """
    This map accumulator rule simply keeps the latest observation and discards the rest
    """
    def __init__(self, source_map_size, world_size_px, world_size_m):
        super(IdentityMapAccumulator, self).__init__(source_map_size, world_size_px, world_size_m)
        self.child_transformer = MapTransformerBase(source_map_size, world_size_px, world_size_m)

    def reset(self):
        super(IdentityMapAccumulator, self).reset()
        self.child_transformer.reset()

    def cuda(self, device=None):
        MapTransformerBase.cuda(self, device)
        self.child_transformer.cuda(device)
        return self

    def init_weights(self):
        pass

    def forward(self, current_maps, coverages, cam_poses, add_mask=None, show=""):
        batch_size = len(cam_poses)

        assert add_mask is None or add_mask[0] is not None, "The first observation in a sequence needs to be used!"

        # If we don't have masked observations, just return each timestep observations
        if add_mask is None:
            self.set_maps(current_maps, cam_poses)
            return current_maps, cam_poses

        maps_r = []

        # If we have masked observations, then for timesteps where observation is masked (False), get the previous observation
        # rotated to the current frame
        for i in range(batch_size):

            # If we don't have a map yet, rotate this observation and initialize a map
            if self.latest_map is None:
                self.set_map(current_maps[i:i + 1], cam_poses[i:i + 1])
                map_g, _ = self.get_map(None)
                self.set_map(map_g, None)

            # Allow masking of observations
            if add_mask is None or add_mask[i]:
                # Transform the observation into the global (map) frame
                self.child_transformer.set_map(current_maps[i:i + 1], cam_poses[i:i + 1])
                obs_g, _ = self.child_transformer.get_map(None)

                # Remember this new map
                self.set_map(obs_g, None)

            # Return this map in the camera frame of reference
            map_r, _ = self.get_map(cam_poses[i:i + 1])

            if show != "":
                Presenter().show_image(map_r.data[0, 0:3], show, torch=True, scale=8, waitkey=1)

            maps_r.append(map_r)

        maps_r = torch.cat(maps_r, dim=0)
        self.set_maps(maps_r, cam_poses)

        return maps_r, cam_poses