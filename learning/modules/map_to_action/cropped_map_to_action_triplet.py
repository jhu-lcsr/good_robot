import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from learning.modules.downsample_map.downsample_res import DownsampleResidual
from learning.inputs.common import empty_float_tensor, cuda_var
from learning.modules.blocks import DenseMlpBlock2
from learning.modules.gs_fpv.recurrent_embedding import RecurrentEmbedding

from visualization import Presenter
import rollout.run_metadata as run_md

HIDDEN_SIZE = 32
RECURRENCE_SIZE = 8
#downsample_factor = 2


class CroppedMapToActionTriplet(nn.Module):
    def __init__(self, map_channels=1, map_size=32,
                 manual=False, path_only=False, recurrence=False):
        super(CroppedMapToActionTriplet, self).__init__()

        self.map_channels = map_channels
        self.manual = manual
        self.path_only = path_only
        self.use_recurrence = recurrence
        if self.path_only:
            print("WARNING! MAP TO ACTION WILL DISCARD GOAL-PREDICTION")

        # Downsample the map to get something suitable for feeding into the perceptron
        #self.downsample = DownsampleResidual(map_channels, factor=downsample_factor)

        self.l_crop = 26
        self.t_crop = 26
        self.r_crop = 38
        self.b_crop = 38
        self.crop_h = 12
        self.crop_w = 12

        map_size_s = self.crop_h * self.crop_w
        map_size_flat = map_size_s * map_channels

        self.recurrence = nn.LSTMCell(4, RECURRENCE_SIZE)
        self.last_h = None
        self.last_c = None

        # Apply the perceptron to produce the action
        mlp_in_size = map_size_flat + RECURRENCE_SIZE# + other_features_size
        self.mlp = DenseMlpBlock2(mlp_in_size, HIDDEN_SIZE, 7)

        self.dropout = nn.Dropout(0.5)

    def forget_recurrence(self):
        self.last_h = torch.zeros([1, RECURRENCE_SIZE]).to(next(self.parameters()).device)
        self.last_c = torch.zeros([1, RECURRENCE_SIZE]).to(next(self.parameters()).device)

    def reset(self):
        if self.use_recurrence:
            self.forget_recurrence()

    def init_weights(self):
        #self.downsample.init_weights()
        self.mlp.init_weights()

    def forward_one(self, maps_r, other_features, firstseg=None):
        # TODO: Log this somewhere
        if self.map_channels < maps_r.size(1):
            maps_r = maps_r[:, 0:self.map_channels]

        if self.manual:
            max, argmax = torch.max(maps_r[:, 1])
            print(argmax)
        if True:
            maps_s = maps_r[:, :, self.t_crop:self.b_crop, self.l_crop:self.r_crop].contiguous()

            # maps_s = self.downsample(maps_r)
            if self.path_only:
                # Copy over the trajectory channel, discarding the goal
                maps_in = torch.zeros_like(maps_s)
                maps_in[:, 0] = maps_s[:, 0]
            else:
                maps_in = maps_s

            DBG = run_md.IS_ROLLOUT# or True# and False
            if DBG:
                for i in range(len(maps_s)):
                    Presenter().show_image(maps_in.data[i], "d", torch=True, waitkey=1, scale=8)

            map_features = maps_in.view([maps_s.size(0), -1])

            # other_features_zero = torch.zeros_like(other_features)
            # mlp_in_features = torch.cat([map_features, other_features_zero], dim=1)

            mlp_in_features = map_features
            if self.use_recurrence:
                if firstseg:
                    self.forget_recurrence()
                hist_features = self.last_h
            else:
                hist_features = torch.zeros([maps_s.size(0), RECURRENCE_SIZE]).to(next(self.parameters()).device)

            mlp_in_features = torch.cat([mlp_in_features, hist_features], dim=1)
            mlp_in_features = self.dropout(mlp_in_features)
            actions_pred = self.mlp(mlp_in_features)

            if self.use_recurrence:
                self.last_h, self.last_c = self.recurrence(actions_pred, (self.last_h, self.last_c))

            # this must be in 0-1 range for BCE loss
            actions_pred = actions_pred.clone()
            actions_pred[:, 3] = torch.sigmoid(actions_pred[:, 3])
            return actions_pred

    def forward(self, maps_r, other_features, fistseg_mask=None):
        # The old PVNv1 doesn't use outer probabilities
        maps_r = maps_r.inner_distribution

        # If we are using recurrence, we can't batch over the sequence. Apply this rule sequentially
        if self.use_recurrence:
            raise NotImplementedError("Recurrence no longer implemented")
            all_act_pred = []
            for i in range(maps_r.size(0)):
                msk_in = fistseg_mask[i] if fistseg_mask is not None else None
                ofin = other_features[i:i+1] if other_features is not None else None
                act_pred_i = self.forward_one(maps_r[i:i+1], ofin, msk_in)
                all_act_pred.append(act_pred_i)
            return torch.cat(all_act_pred, dim=0)

        else:
            return self.forward_one(maps_r, other_features, firstseg=None)
