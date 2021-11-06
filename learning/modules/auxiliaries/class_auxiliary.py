from torch import nn as nn
import torch

from env_config.definitions.landmarks import get_landmark_index_to_name
from data_io.weights import enable_weight_saving
from learning.modules.auxiliary_objective_base import AuxiliaryObjective
from learning.meters_and_metrics.meter_server import log_value
from learning.meters_and_metrics.moving_average import MovingAverageMeter

DBG = True


class ClassAuxiliary(AuxiliaryObjective):
    def __init__(self, name, feature_vec_len=32, num_classes=2, num_outputs=1, *inputs):
        super(ClassAuxiliary, self).__init__(name, *inputs)
        self.channels_in = feature_vec_len
        self.num_classes = num_classes
        self.num_outputs = num_outputs

        self.cls_linear = nn.Linear(feature_vec_len, num_classes * num_outputs)
        enable_weight_saving(self.cls_linear, "aux_class_linear_" + name)

        self.loss = nn.CrossEntropyLoss()
        self.meter_accuracy = MovingAverageMeter(10)

    def cuda(self, device=None):
        AuxiliaryObjective.cuda(self, device)
        return self

    def forward(self, fvectors, labels):

        fvectors = torch.cat(fvectors, dim=0)
        labels = torch.cat(labels, dim=0)

        pred = self.cls_linear(fvectors)

        # If we're predicting multiple outputs for each input, reshape accordingly.
        pred = pred.view([-1, self.num_classes])
        labels = labels.view([-1])

        loss = self.loss(pred, labels)

        _, maxidx = torch.max(pred, 1)
        correct = torch.sum((maxidx == labels).long()).data.item()
        accuracy = correct / len(labels)

        self.meter_accuracy.put(accuracy)
        log_value(self.name + "/accuracy", self.meter_accuracy.get())

        # TODO: Track accuracy
        return loss, 1