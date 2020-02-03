import struct
import math
import numpy as np
import warnings
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import ndimage


# Cross entropy loss for 2D outputs
class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def action_space_argmax(primitive_action, push_predictions, grasp_predictions, place_predictions=None):
    # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
    each_action_max_coordinate = {
        'push': np.unravel_index(np.ma.argmax(push_predictions), push_predictions.shape), # push, index 0
        'grasp': np.unravel_index(np.ma.argmax(grasp_predictions), grasp_predictions.shape),
    }
    each_action_predicted_value = {
        'push': push_predictions[each_action_max_coordinate['push']], # push, index 0
        'grasp': grasp_predictions[each_action_max_coordinate['grasp']],
    }
    if place_predictions is not None:
        each_action_max_coordinate['place'] = np.unravel_index(np.ma.argmax(place_predictions), place_predictions.shape)
        each_action_predicted_value['place'] = place_predictions[each_action_max_coordinate['place']]
    # we will actually execute the best pixel index of the selected action
    best_pixel_index = each_action_max_coordinate[primitive_action]
    predicted_value = each_action_predicted_value[primitive_action]
    return best_pixel_index, each_action_max_coordinate, predicted_value
    