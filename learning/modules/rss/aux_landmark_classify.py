import torch
import torch.nn.functional as F
import torch.nn as nn

from learning.modules.module_base import ModuleBase
from learning.modules.gather_2d import Gather2D


class AuxLandmarkClassifier(ModuleBase):

    def __init__(self, feature_vec_len, num_outputs=63):
        super(AuxLandmarkClassifier, self).__init__()
        self.aux_class_linear = nn.Linear(feature_vec_len, num_outputs)
        self.gather_2d = Gather2D()
        self.aux_loss = nn.CrossEntropyLoss(reduce=False, size_average=False)

    def init_weights(self):
        self.aux_class_linear.weight.data.normal_(0, 0.001)
        self.aux_class_linear.bias.data.fill_(0)

    def loss(self, pred, landmark_labels):
        return self.aux_loss(pred, landmark_labels)

    def forward(self, feature_images, landmark_coords):
        landmark_feature_vectors = self.gather_2d(feature_images, landmark_coords)
        pred = self.aux_class_linear(landmark_feature_vectors)
        return pred