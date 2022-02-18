import torch
from torch import nn as nn

from learning.modules.blocks import ResBlockStrided, ResBlock


class ResNet15(torch.nn.Module):
    def __init__(self, channels):
        super(ResNet15, self).__init__()

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(3, channels, 3, stride=1)

        self.block1 = ResBlockStrided(channels, stride=2)
        self.block15 = ResBlock(channels)
        self.block2 = ResBlockStrided(channels, stride=2)
        self.block25 = ResBlock(channels)
        self.block3 = ResBlockStrided(channels, stride=2)
        self.block35 = ResBlock(channels)
        self.block4 = ResBlockStrided(channels, stride=2)
        self.block45 = ResBlock(channels)

        self.res_norm = nn.InstanceNorm2d(channels)

    def init_weights(self):
        self.block1.init_weights()
        self.block2.init_weights()
        self.block3.init_weights()
        self.block4.init_weights()

        self.block15.init_weights()
        self.block25.init_weights()
        self.block35.init_weights()
        self.block45.init_weights()

        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)

    def forward(self, input):
        x = self.conv1(input)
        x = self.block1(x)
        x = self.block15(x)
        x = self.block2(x)
        x = self.block25(x)
        x = self.block3(x)
        x = self.block35(x)
        x = self.block4(x)
        x = self.block45(x)
        x = self.res_norm(x)
        return x


