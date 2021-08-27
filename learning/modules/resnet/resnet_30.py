import torch
from torch import nn as nn

from learning.modules.blocks import ResBlockStrided, ResBlock


class ResNet30(torch.nn.Module):
    def __init__(self, channels, down_pad=False):
        super(ResNet30, self).__init__()

        down_padding = 0
        if down_pad:
            down_padding = 1

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(3, channels, 3, stride=1, padding=down_padding)

        self.block1 = ResBlock(channels)
        self.block2 = ResBlock(channels)
        self.block3 = ResBlock(channels)
        self.block4 = ResBlockStrided(channels, stride=2, down_padding=down_padding)

        self.block5 = ResBlock(channels)
        self.block6 = ResBlock(channels)
        self.block7 = ResBlock(channels)
        self.block8 = ResBlockStrided(channels, stride=2, down_padding=down_padding)

        self.block9 = ResBlock(channels)
        self.block10 = ResBlock(channels)
        self.block11 = ResBlock(channels)
        self.block12 = ResBlockStrided(channels, stride=2, down_padding=down_padding)

        self.res_norm = nn.InstanceNorm2d(channels)

    def init_weights(self):
        self.block1.init_weights()
        self.block2.init_weights()
        self.block3.init_weights()
        self.block4.init_weights()
        self.block5.init_weights()
        self.block6.init_weights()
        self.block7.init_weights()
        self.block8.init_weights()
        self.block9.init_weights()
        self.block10.init_weights()
        self.block11.init_weights()
        self.block12.init_weights()

        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)

    def forward(self, input):
        x = self.conv1(input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.res_norm(x)
        return x


