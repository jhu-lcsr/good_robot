import torch
import torch.nn.functional as F
import torch.nn as nn

from learning.modules.downsample_map.downsample_res import DownsampleResidual
from learning.modules.blocks import DenseMlpBlock2

HIDDEN_SIZE = 64
downsample_factor = 2


class EgoMapToActionTriplet(nn.Module):
    def __init__(self, map_channels=1, map_size=32, other_features_size=120):
        super(EgoMapToActionTriplet, self).__init__()

        self.map_channels = map_channels

        # Downsample the map to get something suitable for feeding into the perceptron
        self.downsample = DownsampleResidual(map_channels, factor=downsample_factor)

        map_size_s = int(map_size / downsample_factor)

        # Apply the perceptron to produce the action
        map_size_flat = map_size_s * map_size_s * map_channels
        mlp_in_size = map_size_flat# + other_features_size
        self.mlp = DenseMlpBlock2(mlp_in_size, HIDDEN_SIZE, 4)

        self.dropout = nn.Dropout(0.5)

    def init_weights(self):
        self.downsample.init_weights()
        self.mlp.init_weights()

    def forward(self, maps_r, other_features):

        # TODO: Log this somewhere
        if self.map_channels < maps_r.size(1):
            maps_r = maps_r[:, 0:self.map_channels]

        maps_s = self.downsample(maps_r)

        map_features = maps_s.view([maps_s.size(0), -1])

        #other_features_zero = torch.zeros_like(other_features)
        #mlp_in_features = torch.cat([map_features, other_features_zero], dim=1)

        mlp_in_features = map_features
        mlp_in_features = self.dropout(mlp_in_features)

        actions_pred = self.mlp(mlp_in_features)

        # this must be in 0-1 range for BCE loss
        actions_pred[:,3] = torch.sigmoid(actions_pred[:,3])

        return actions_pred