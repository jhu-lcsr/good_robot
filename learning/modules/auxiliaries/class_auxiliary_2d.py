from data_io.weights import enable_weight_saving
from learning.modules.auxiliary_objective_base import AuxiliaryObjective
from learning.modules.gather_2d import Gather2D

import torch
from torch import nn as nn

from learning.modules.auxiliary_objective_base import AuxiliaryObjective
from learning.meters_and_metrics.moving_average import MovingAverageMeter
from learning.meters_and_metrics.meter_server import log_value

from visualization import Presenter
from transformations import pos_m_to_px

DBG = False


class ClassAuxiliary2D(AuxiliaryObjective):
    def __init__(self, name, feature_vec_len=32, num_classes=64, dropout=0, *inputs):
        super(ClassAuxiliary2D, self).__init__(name, *inputs)
        self.gather_2d = Gather2D()
        self.channels_in = feature_vec_len
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes

        self.cls_linear = nn.Linear(feature_vec_len, num_classes)
        enable_weight_saving(self.cls_linear, "aux_class_linear_2d_" + name)
        self.loss = nn.CrossEntropyLoss()
        if self.name == "aux_grounding_map":
            self.loss.weight = torch.tensor([0.5, 0.5])
        self.meter_accuracy = MovingAverageMeter(10)

    def cuda(self, device=None):
        AuxiliaryObjective.cuda(self, device)
        return self

    def plot_pts(self, image, pts):
        """
        :param image: CxHxW image
        :param pts: Nx2 points - (H,W) coords in the image
        :return:
        """
        image = image.cpu().data.numpy()
        image = image.transpose((1, 2, 0))
        pts = pts.cpu().data.numpy()
        image[:, :, 0] = 0.0
        for pt in pts:
            image[pt[0], pt[1], 0] = 1.0

        Presenter().show_image(image[:, :, 0:3], f"aux_class_2d:{self.name}", torch=False, waitkey=1, scale=8)

    def forward(self, images, lm_pos_list_px, lm_indices_list, tensor_store=None):
        """
        :param images:
        :param lm_pos_list_px:
        :param lm_indices_list:
        :param tensor_store: KeyTensorStore where to store computed predictions for visualization and stuff
        :return:
        """

        batch_size = len(images)

        if type(images) == list:
            images = torch.cat(images, dim=0)

        images = self.dropout(images)

        # Take the first N channels if we have more channels available than necessary
        if images.size(1) > self.channels_in:
            images = images[:, 0:self.channels_in, :, :]

        loss_out = None
        accuracy = 0

        # Move the channel dimension to the back to apply the linear layer, and then back forward to the channel slot
        all_predictions = self.cls_linear(images.permute((0, 2, 3, 1))).permute(0, 3, 1, 2)

        if tensor_store is not None:
            tensor_store.keep_inputs(f"{self.name}_predictions", all_predictions)

        for i in range(batch_size):
            # Apply the 2D gather to get a batch of feature vectors extracted at the landmark positions
            pred_i = all_predictions[i:i+1]
            lm_pos_i = lm_pos_list_px[i]
            if lm_pos_i is None:
                break

            # TODO: Very careful! This one flips the lm_pos_i axis and does a conversion
            #if self.world_size_px is not None:
            #    lm_pos_i = pos_m_to_px(lm_pos_i, self.world_size_px).long().unsqueeze(0)
            predictions = self.gather_2d(pred_i, lm_pos_i)

            if DBG and i == batch_size - 1:
                if pred_i.shape[3] > 32:
                    plot_pred = pred_i[0, [0,3,24], :, :]
                else:
                    plot_pred = pred_i[0, 0:3, :, :]

                self.plot_pts(plot_pred, lm_pos_i[0:1])

            #predictions = self.cls_linear(feature_vectors)
            labels = lm_indices_list[i]
            if len(labels.shape) == 0:
                labels = labels.unsqueeze(0)

            # The batch here is over the landmarks visible in the image
            loss = self.loss(predictions, labels)
            if loss_out is None:
                loss_out = loss
            else:
                loss_out += loss

            _, pred_idx = torch.max(predictions.data, 1)
            correct = torch.sum((pred_idx == labels.data).float())
            total = float(len(labels))
            accuracy = (correct / total)

            self.meter_accuracy.put(accuracy)

        #
        if loss_out is None:
            loss_out = 0
        #else:
        #    loss_out /= batch_size
        acc = self.meter_accuracy.get()
        # This acc * batch_size is a hack to get around auxiliary_losses later dividing by batch_size
        return loss_out, {"accuracy": acc * batch_size}, batch_size
