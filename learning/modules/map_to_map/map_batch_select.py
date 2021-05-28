import torch
import numpy as np
import torch.nn as nn

class MapBatchSelect(nn.Module):
    """
    Given a batch of B maps and poses, and a boolean mask of length B, return a batch of P maps and poses, where
    P is the number of True in the boolean mask.

    This is used to pick a subset of semantic maps for path-prediction, if we are not planning on every single timestep
    """

    def __init__(self):
        super(MapBatchSelect, self).__init__()

    def init_weights(self):
        pass

    def one(self, tensor, plan_mask, device):
        mask_t = torch.Tensor(plan_mask) == True
        mask_t = mask_t.to(device)
        return tensor[mask_t]

    def forward(self, maps, map_coverages, map_poses, cam_poses, noisy_poses, start_poses, sent_embeds, plan_mask=None, show=""):
        if plan_mask is None:
            return maps, map_coverages, map_poses, cam_poses, noisy_poses, start_poses, sent_embeds

        mask_t = torch.Tensor(plan_mask) == True
        mask_t = mask_t.to(maps.device)

        maps_size = list(maps.size())[1:]
        select_maps = maps[mask_t[:, np.newaxis, np.newaxis, np.newaxis].expand_as(maps)].view([-1] + maps_size)
        covs_size = list(map_coverages.size())[1:]
        select_coverages = map_coverages[mask_t[:, np.newaxis, np.newaxis, np.newaxis].expand_as(map_coverages)].view([-1] + covs_size)
        if sent_embeds.shape[0] == mask_t.shape[0]:
            select_sent_embeds = sent_embeds[mask_t[:, np.newaxis].expand_as(sent_embeds)].view([-1, sent_embeds.size(1)])
        else:
            select_sent_embeds = sent_embeds
        select_poses = map_poses[mask_t] if map_poses is not None else None
        select_cam_poses = cam_poses[mask_t] if cam_poses is not None else None
        select_noisy_poses = noisy_poses[mask_t] if noisy_poses is not None else None
        select_start_poses = start_poses[mask_t] if start_poses is not None else None

        #print("Selected " + str(len(select_maps)) + " maps from " + str(len(maps)))

        return select_maps, select_coverages, select_poses, select_cam_poses, select_noisy_poses, select_start_poses, select_sent_embeds