import torch
from torch import nn as nn

from learning.modules.blocks import ResBlockStrided

class ResNet9Stride32(nn.Module):
    def __init__(self, in_channels, channels, down_pad=True):
        super(ResNet9Stride32, self).__init__()

        down_padding = 0
        if down_pad:
            down_padding = 1

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=2, padding=down_padding)
        self.block1 = ResBlockStrided(channels, stride=2, down_padding=down_padding)
        self.block2 = ResBlockStrided(channels, stride=2, down_padding=down_padding)
        self.block3 = ResBlockStrided(channels, stride=2, down_padding=down_padding)
        self.block4 = ResBlockStrided(channels, stride=2, down_padding=down_padding)

        self.res_norm = nn.InstanceNorm2d(channels)

    def get_downscale_factor(self):
        return 32

    def init_weights(self):
        self.block1.init_weights()
        self.block2.init_weights()
        self.block3.init_weights()
        self.block4.init_weights()
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.conv1.bias.data.fill_(0)

    def forward(self, input):
        x = self.conv1(input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
