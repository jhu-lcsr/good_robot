import torch
import torch.nn as nn
from torch.autograd import Variable

from learning.inputs.sequence import mask_tensors
from learning.inputs.common import empty_float_tensor


class SpatialSoftmax2d(nn.Module):

    def __init__(self, log=False):
        super(SpatialSoftmax2d, self).__init__()
        if log:
            self.softmax = nn.LogSoftmax()
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        batch_size = images.size(0)
        num_channels = images.size(1)
        height = images.size(2)
        width = images.size(3)

        images = images.view([batch_size * num_channels, width * height])
        img_out = self.softmax(images)
        img_out = img_out.view([batch_size, num_channels, height, width])

        return img_out