import torch
import numpy as np
import torch.nn as nn

from visualization import Presenter

class GoalPredictionGoodCriterion(nn.Module):
    """
    This module takes a given goal-prediction mask and a correct goal location mask and
    finds whether the argmax location in the predicted mask is close to the argmax location in the ground truth mask
    On the trajectory-prediction + control model (CoRL), this is used in the following way:
        If the goal prediction is good, we train the controller to execute the trajectory
        If the goal prediction is bad, we skip the gradient update
    """
    def __init__(self, ok_distance=3.2):
        super(GoalPredictionGoodCriterion, self).__init__()
        self.ok_distance = ok_distance

    def forward(self, masks, goal_pos):

        masks = torch.cat([m.inner_distribution for m in masks], dim=0)

        if masks.size(1) == 1:
            return False

        # TODO: Handle batches if necessary
        goal_mask = masks[0, 1, :, :]
        goal_pos = goal_pos[0]
        goal_mask_flat = goal_mask.view([1, -1])
        max_val, argmax = goal_mask_flat.max(1)
        argmax_loc_x = argmax / goal_mask.size(1)
        argmax_loc_y = torch.remainder(argmax, goal_mask.size(1))
        argmax_loc = torch.cat([argmax_loc_x.unsqueeze(1), argmax_loc_y.unsqueeze(1)], 1).double()

        dist = (argmax_loc - goal_pos).float().norm(dim=1)
        success = dist < self.ok_distance

        return success