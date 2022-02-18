import torch
from torch import nn as nn

"""
class CudaModule(torch.nn.Module):
    def __init__(self):
        super(CudaModule, self).__init__()

        self.is_cuda = False
        self.cuda_device = None

    def cuda(self, device=None):
        nn.Module.cuda(self, device)
        self.is_cuda = True
        self.cuda_device = device
        return self
"""