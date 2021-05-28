from data_io.weights import enable_weight_saving

from learning.inputs.common import empty_float_tensor, cuda_var

from learning.modules.auxiliary_objective_base import AuxiliaryObjective
from learning.modules.gather_2d import Gather2D
from learning.meters_and_metrics.moving_average import MovingAverageMeter
from learning.meters_and_metrics.meter_server import log_value

import torch
from torch import nn as nn

from learning.modules.auxiliary_objective_base import AuxiliaryObjective
from visualization import Presenter

DBG = False


class GoalAuxiliary2D(AuxiliaryObjective):
    def __init__(self, name, channels_in=32, map_size_px=32, *inputs):
        super(GoalAuxiliary2D, self).__init__(name, *inputs)
        self.gather_2d = Gather2D()

        self.map_size_px = map_size_px
        self.channels_in = channels_in
        self.goal_linear = nn.Linear(channels_in, 2)
        enable_weight_saving(self.goal_linear, "aux_goal_linear_" + name)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy_meter = MovingAverageMeter(10)

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
        image = image.transpose((1,2,0))
        pts = pts.cpu().data.numpy()
        image[:, :, 1] = 0.0
        for pt in pts:
            image[pt[0], pt[1], 0] = 1.0

        Presenter().show_image(image[:,:,0:3], "aux_goal_" + self.name, torch=False, waitkey=True, scale=8)

    def forward(self, map, goal_pos):
        batch_size = len(map)

        map = torch.cat(map, dim=0)

        if map.size(1) > self.channels_in:
            map = map[:, 0:self.channels_in, :, :]

        loss_out = None
        for i in range(batch_size):

            goal_coords_in_map = goal_pos[i].long()
            map_i = map[i:i+1]

            #goal_coords_in_map = pos_m_to_px(goal_pos_i, self.map_size_px, self.world_size_m, self.world_size_px).long()
            neg_samples = 1
            neg_coords_size = list(goal_coords_in_map.size())
            neg_coords_size[0] = neg_coords_size[0] * neg_samples
            all_coords_size = list(goal_coords_in_map.size())
            all_coords_size[0] += neg_coords_size[0]

            goal_negative_coords_in_map = empty_float_tensor(neg_coords_size)
            range_min = 0
            range_max = self.map_size_px
            goal_negative_coords_in_map.uniform_(range_min, range_max)
            goal_negative_coords_in_map = goal_negative_coords_in_map.long().to(goal_coords_in_map.device)

            sample_pt_coords = torch.cat([goal_coords_in_map, goal_negative_coords_in_map], dim=0).long()
            sample_pt_labels = torch.zeros([all_coords_size[0]]).long().to(goal_coords_in_map.device)
            sample_pt_labels[0] = 1
            sample_pt_labels[1:] = 0

            sample_pt_features = self.gather_2d(map_i, sample_pt_coords)

            if DBG:
                self.plot_pts(map[0], sample_pt_coords)

            pt_predictions = self.goal_linear(sample_pt_features)
            aux_loss_goal = self.loss(pt_predictions, sample_pt_labels)

            _, pred_idx = torch.max(pt_predictions.data, 1)
            correct = torch.sum((pred_idx == sample_pt_labels.data).long())
            total = float(len(sample_pt_labels))
            accuracy = float(correct.item()) / total
            self.accuracy_meter.put(accuracy)
            log_value(self.name + "/accuracy", self.accuracy_meter.get())

            if loss_out is None:
                loss_out = aux_loss_goal
            else:
                loss_out += aux_loss_goal

            # TODO: Consider batch size / count

        return loss_out, batch_size