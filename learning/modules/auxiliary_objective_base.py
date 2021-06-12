import torch
from torch import nn as nn


class AuxiliaryObjective(nn.Module):
    def __init__(self, name, *inputs):
        super(AuxiliaryObjective, self).__init__()
        self.required_inputs = inputs
        self.name = name

    def get_name(self):
        return self.name

    def get_required_inputs(self):
        return self.required_inputs
