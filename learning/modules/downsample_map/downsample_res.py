import torch
from torch import nn as nn
import torch.nn.functional as F

from learning.modules.resnet.resnet_13_light import ResNet13Light
from learning.modules.blocks import ResBlockStrided


class DownsampleResidual(torch.nn.Module):
    """
    Fun class that will repeatedly apply residual blocks with strided convolutions until the
    input image is downsized by the given factor, which must be one of (2, 4, 8, 16).

    """
    def __init__(self, channels=32, factor=4):
        super(DownsampleResidual, self).__init__()
        pad=1
        self.factor = factor
        self.channels = channels

        if factor >= 2:
            self.res2 = ResBlockStrided(channels, stride=2, down_padding=pad, nonorm=False)
        if factor >= 4:
            self.res4 = ResBlockStrided(channels, stride=2, down_padding=pad, nonorm=False)
        if factor >= 8:
            self.res8 = ResBlockStrided(channels, stride=2, down_padding=pad, nonorm=False)
        if factor >= 16:
            self.res16 = ResBlockStrided(channels, stride=2, down_padding=pad, nonorm=False)

        self.res_norm = nn.InstanceNorm2d(channels)

    def init_weights(self):
        if self.factor >= 2:
            self.res2.init_weights()
        if self.factor >= 4:
            self.res4.init_weights()
        if self.factor >= 8:
            self.res8.init_weights()
        if self.factor >= 16:
            self.res16.init_weights()

    def forward(self, image):
        x = image
        if self.factor >= 2:
            x = self.res2(x)
        if self.factor >= 4:
            x = self.res4(x)
        if self.factor >= 8:
            x = self.res8(x)
        if self.factor >= 16:
            x = self.res16(x)
        return x