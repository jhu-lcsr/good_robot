from learning.modules.auxiliary_objective_base import AuxiliaryObjective
from learning.modules.gather_2d import Gather2D

import torch
from torch import nn as nn

from learning.modules.auxiliary_objective_base import AuxiliaryObjective

from visualization import Presenter
from transformations import pos_m_to_px

DBG = False


class FeatureRegularizationAuxiliary2D(AuxiliaryObjective):
    def __init__(self, name, kind="l1", *inputs):
        super(FeatureRegularizationAuxiliary2D, self).__init__(name, *inputs)
        self.kind = kind

    def cuda(self, device=None):
        AuxiliaryObjective.cuda(self, device)
        return self

    def forward(self, images):
        images = torch.cat(images, dim=0)
        images_abv = torch.abs(images)
        loss = torch.mean(images_abv)
        return loss, 1