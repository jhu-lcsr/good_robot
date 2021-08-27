import torch
from torch import nn as nn

from learning.modules.resnet.resnet_13 import ResNet13


class ResNet13Comp(torch.nn.Module):
    def __init__(self, res_channels, map_channels):
        super(ResNet13Comp, self).__init__()

        self.resnet = ResNet13(res_channels)
        self.conv_end = nn.Conv2d(res_channels, map_channels, 1, stride=1, padding=0, bias=True)
        self.norm = nn.InstanceNorm2d(map_channels)

    def init_weights(self):
        self.resnet.init_weights()
        torch.nn.init.kaiming_uniform(self.conv_end.weight)
        self.conv_end.bias.data.fill_(0)

    def forward(self, input):
        x = self.resnet(input)
        x = self.conv_end(x)
        x = self.norm(x)
        return x


