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

def demo_space_argmax(primitive_action, demo_best_pix_ind, push_predictions, grasp_predictions, place_predictions=None):
    # TODO(adit98) add code to do l2 mask comparison here eventually (move from main)

    # modify demo_best_pix_ind to have rotation ind 0 (since we only run the best rotation)
    demo_best_pix_ind = (0, demo_best_pix_ind[1], demo_best_pix_ind[2])

    # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
    each_action_max_coordinate = {
        'push': demo_best_pix_ind,
        'grasp': demo_best_pix_ind,
    }
    each_action_predicted_value = {
        'push': push_predictions[demo_best_pix_ind], # push, index 0
        'grasp': grasp_predictions[demo_best_pix_ind],
    }
    if place_predictions is not None:
        each_action_max_coordinate['place'] = demo_best_pix_ind,
        each_action_predicted_value['place'] = place_predictions[demo_best_pix_ind]

    # we will actually execute the best pixel index of the selected action
    best_pixel_index = each_action_max_coordinate[primitive_action]
    predicted_value = each_action_predicted_value[primitive_action]
    return best_pixel_index, each_action_max_coordinate, predicted_value

def random_unmasked_index_in_mask_array(maskarray):
    """ Return an index in a masked array which is selected with a uniform random distribution from the valid aka unmasked entries where the masked value is 0.
    """
    # TODO(ahundt) currently a whole new float mask is created to define the probabilities. There may be a much more efficient way to handle this.
    if np.ma.is_masked(maskarray):
        # Randomly select from only regions which are valid exploration regions
        p = (np.array(1-maskarray.mask, dtype=np.float)/np.float(maskarray.count())).ravel()
    else:
        # Uniform random across all locations
        p = None

    return np.unravel_index(np.random.choice(maskarray.size, p=p), maskarray.shape)


def action_space_explore_random(primitive_action, push_predictions, grasp_predictions, place_predictions=None):
    """ Return an index in a masked prediction arrays which is selected with a uniform random distribution from the valid aka unmasked entries where the masked value is 0. (rotation, y, x)
    """
    each_action_rand_coordinate = {
        'push': random_unmasked_index_in_mask_array(push_predictions), # push, index 0
        'grasp': random_unmasked_index_in_mask_array(grasp_predictions),
    }
    each_action_predicted_value = {
        'push': push_predictions[each_action_rand_coordinate['push']], # push, index 0
        'grasp': grasp_predictions[each_action_rand_coordinate['grasp']],
    }
    if place_predictions is not None:
        each_action_rand_coordinate['place'] = random_unmasked_index_in_mask_array(place_predictions)
        each_action_predicted_value['place'] = place_predictions[each_action_rand_coordinate['place']]
    # we will actually execute the best pixel index of the selected action
    best_pixel_index = each_action_rand_coordinate[primitive_action]
    predicted_value = each_action_predicted_value[primitive_action]
    return best_pixel_index, each_action_rand_coordinate, predicted_value
