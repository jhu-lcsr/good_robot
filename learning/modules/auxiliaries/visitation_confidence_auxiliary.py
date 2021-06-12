from torch import nn as nn
import torch

from learning.modules.auxiliary_objective_base import AuxiliaryObjective
from learning.meters_and_metrics.moving_average import MovingAverageMeter

DBG = True

from visualization import Presenter

class VisitationConfidenceAuxiliary(AuxiliaryObjective):
    def __init__(self, name, world_size_px, *inputs):
        super(VisitationConfidenceAuxiliary, self).__init__(name, *inputs)
        self.loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.world_size_px = world_size_px
        self.acc_threshold = float(self.world_size_px) / 10
        #self.loss = nn.CrossEntropyLoss()
        self.meter_accuracy = MovingAverageMeter(10)

    def cuda(self, device=None):
        AuxiliaryObjective.cuda(self, device)
        return self

    def forward(self, confidence_scores, v_dist, v_dist_gt):
        """
        :param confidence_scores:
        :param v_dist: List of 1x2x64x64 tensors
        :param v_dist_gt: List of 1x2x64x64 tensors
        :return:
        """
        v_dist = torch.cat(v_dist, dim=0)

        #for vd in v_dist:
        #    Presenter().show_image(vd.detach().cpu(), "v_dist_pred", scale=2, waitkey=True)

        v_dist_gt = torch.cat(v_dist_gt, dim=0)
        confidences = torch.cat(confidence_scores, dim=0)
        goal_confidence = confidences[:, 0]

        # Grab the goal channel
        g_dist = v_dist[:, 1, :, :]
        g_dist_gt = v_dist_gt[:, 1, :, :]
        batch_size = v_dist.shape[0]

        _, argmax_gt_goal = g_dist_gt.view(batch_size, -1).max(1)
        gt_goal_pos_x = argmax_gt_goal / g_dist_gt.shape[1]
        gt_goal_pos_y = argmax_gt_goal % g_dist_gt.shape[1]
        gt_goal_pos = torch.stack([gt_goal_pos_x, gt_goal_pos_y], dim=1)

        _, argmax_pred_goal = g_dist.view(batch_size, -1).max(1)
        pred_goal_pos_x = argmax_pred_goal / g_dist.shape[1]
        pred_goal_pos_y = argmax_pred_goal % g_dist.shape[1]
        pred_goal_pos = torch.stack([pred_goal_pos_x, pred_goal_pos_y], dim=1)

        dst_to_best_stop = torch.norm((pred_goal_pos - gt_goal_pos).float(), dim=1)
        # TODO: Grab 3.2 from config
        target = (dst_to_best_stop < self.acc_threshold).long()
        predictions = (goal_confidence > 0.0).long()

        confidence_loss = self.loss(goal_confidence, target.float())
        total_correct = (predictions == target).sum()

        return confidence_loss, {"goal_confidence_accuracy": total_correct}, batch_size
