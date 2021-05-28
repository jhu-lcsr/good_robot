import torch
import torch.nn as nn
from learning.models.semantic_map.map_affine import MapAffine
from visualization import Presenter


class MapTransformer(nn.Module):

    # TODO: Refactor this entire getting/setting idea
    def __init__(self, source_map_size, world_size_px, world_size_m, dest_map_size=None):
        super(MapTransformer, self).__init__()

        if dest_map_size is None:
            dest_map_size = source_map_size

        self.map_affine = MapAffine(
            source_map_size=source_map_size,
            dest_map_size = dest_map_size,
            world_size_px=world_size_px,
            world_size_m = world_size_m)

    def init_weights(self):
        pass

    def forward(self, maps, map_poses, new_map_poses, skip_channel=None):
        if skip_channel is not None:
            total_num_channels = maps.shape[1]
            incl_channels = list(range(total_num_channels))
            incl_channels.remove(skip_channel)
            select_maps = self.map_affine(maps[:, incl_channels, :, :], map_poses, new_map_poses)
            out_channel_list = []
            for i in range(total_num_channels):
                if i in incl_channels:
                    out_channel_list.append(select_maps[:, i, :, :])
                else:
                    out_channel_list.append(maps[:, skip_channel, :, :])
            transformed_maps = torch.stack(out_channel_list, dim=1)

        else:
            transformed_maps = self.map_affine(maps, map_poses, new_map_poses)

        return transformed_maps, new_map_poses