import torch
from torch import nn as nn

from learning.modules.resnet.resnet_13_light import ResNet13Light
from learning.modules.resnet.resnet_13_s import ResNet13S
from learning.modules.cuda_module import CudaModule


class IdentityImgToImg(CudaModule):
    def __init__(self):
        super(IdentityImgToImg, self).__init__()

    def cuda(self, device=None):
        CudaModule.cuda(self, device)

    def init_weights(self):
        pass

    def forward(self, input):
        return input