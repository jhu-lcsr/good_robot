import torch
from torch import nn as nn

from learning.modules.blocks import ResBlock, ResBlockConditional

class ResNetConditional(nn.Module):
    def __init__(self, embed_size, channels, c_out):
        super(ResNetConditional, self).__init__()

        self.block1 = ResBlock(channels)                            # RF: 5x5
        self.block1a = ResBlock(channels)                           # RF: 9x9
        self.cblock1 = ResBlockConditional(embed_size, channels)    # RF: 9x9
        self.block2 = ResBlock(channels)                            # RF: 13x13
        self.block2a = ResBlock(channels)  # RF: 17x17
        self.cblock2 = ResBlockConditional(embed_size, channels)    # RF: 17x17
        self.block3 = ResBlock(channels)                            # RF: 21x21
        self.block3a = ResBlock(channels)  # RF: 25x25
        self.cblock3 = ResBlockConditional(embed_size, channels)    # RF: 25x25
        self.block4 = ResBlock(channels)                            # RF: 29x29
        self.block4a = ResBlock(channels)  # RF: 33x33
        self.cblock4 = ResBlockConditional(embed_size, channels)    # RF: 33x33
        self.block5 = ResBlock(channels)                            # RF: 37x37
        self.block5a = ResBlock(channels)  # RF: 41x41
        self.cblock5 = ResBlockConditional(embed_size, channels)    # RF: 41x41
        self.block6 = ResBlock(channels)                            # RF: 45x45
        self.block6a = ResBlock(channels)  # RF: 49x49
        self.cblock6 = ResBlockConditional(embed_size, channels)    # RF: 49x49
        self.block7 = ResBlock(channels)                            # RF: 53x53
        self.block7a = ResBlock(channels)                            # RF: 57x57
        self.cblock7 = ResBlockConditional(embed_size, channels)    # RF: 57x57
        self.block8 = ResBlock(channels)                            # RF: 61x61
        self.block8a = ResBlock(channels)                            # RF: 65x65
        self.cblock8 = ResBlockConditional(embed_size, channels, c_out)    # RF: 65x65

    def init_weights(self):
        for mod in self.modules():
            if hasattr(mod, "init_weights") and mod is not self:
                mod.init_weights()

    def forward(self, inputs, contexts):
        x = self.block1(inputs)
        x = self.block1a(x)
        x = self.cblock1(x, contexts)
        x = self.block2(x)
        x = self.block2a(x)
        x = self.cblock2(x, contexts)
        x = self.block3(x)
        x = self.block3a(x)
        x = self.cblock3(x, contexts)
        x = self.block4(x)
        x = self.block4a(x)
        x = self.cblock4(x, contexts)
        x = self.block5(x)
        x = self.block5a(x)
        x = self.cblock5(x, contexts)
        x = self.block6(x)
        x = self.block6a(x)
        x = self.cblock6(x, contexts)
        x = self.block7(x)
        x = self.block7a(x)
        x = self.cblock7(x, contexts)
        x = self.block8(x)
        x = self.block8a(x)
        x = self.cblock8(x, contexts)
        return x