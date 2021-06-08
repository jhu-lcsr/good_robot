import torch

from deprecated.learning.models.semantic_map.map_affine_old import Rotation2DRelative
from learning.models.semantic_map.grid_sampler import GridSampler
from learning.modules.map_to_action.ego_map_to_action_triplet import EgoMapToActionTriplet

HIDDEN_SIZE = 32


class GlobalMapToActionTriplet(torch.nn.Module):
    def __init__(self, map_channels, map_size=(30,30), other_features_size=120):
        super(GlobalMapToActionTriplet, self).__init__()

        # Rotate the global map to the robot's current frame
        self.map_affine = Rotation2DRelative()
        self.grid_sampler = GridSampler()

        # Then apply the local map to action rule
        self.ego_map_to_action = EgoMapToActionTriplet(map_channels, map_size, other_features_size)

    def init_weights(self):
        self.ego_map_to_action.init_weights()

    def forward(self, map_g, other_features, pose):
        # TODO: Correctly call this
        # Transform global-frame map to robot egocentric frame
        grid_mapping = self.map_affine(map_g, pose)
        map_r = self.grid_sampler(map_g, grid_mapping)

        # Apply the egocentric map rule
        action_pred = self.ego_map_to_action(map_r, other_features, pose)

        return action_pred