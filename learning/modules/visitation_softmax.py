import torch
import torch.nn as nn
import numpy as np


class VisitationSoftmax(nn.Module):

    def __init__(self, log=False):
        super(VisitationSoftmax, self).__init__()
        self.log = log
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, visitation_distributions, goal_outside_score=None):
        """
        Applies softmax on visitation distributions, while handling the case where we assign additional
        probability of the goal being outside of the observed map region.
        :param visitation_distributions:
        :return: Nx3xHxW tensor where first channel is probability over visited locations, second channel is probability of stop locations,
        third channel is a copy of the same value indicating the probability that goal location is not visible
        """
        batch_size = visitation_distributions.size(0)
        num_channels = visitation_distributions.size(1)
        assert num_channels == 2, "Must have 2 channels: visitation distribution scores and goal distribution scores"
        height = visitation_distributions.size(2)
        width = visitation_distributions.size(3)

        visitation_dist_scores = visitation_distributions[:, 0, :, :]
        goal_inside_dist_scores = visitation_distributions[:, 1, :, :]

        softmax_func = self.log_softmax if self.log else self.softmax

        # Visitation distribution: Flatten, softmax, reshape back
        visitation_dist = softmax_func(visitation_dist_scores.view(batch_size, width*height)).view(visitation_dist_scores.size())

        # We are modelling OOB probability
        if goal_outside_score is not None:
            # Goal distribution: Flatten, append outside score, softmax, split off outside score, reshape back
            goal_scores_full = torch.cat([goal_inside_dist_scores.view(batch_size, width*height),goal_outside_score[:, np.newaxis]], dim=1)
            goal_dist_full = softmax_func(goal_scores_full)
            goal_inside_partial_dist = goal_dist_full[:, :-1].view(goal_inside_dist_scores.size())
            goal_outside_prob_or_logprob = goal_dist_full[:, -1]

            # Re-assemble back into the Bx2xHxW tensor representation
            visitation_prob_or_log_prob_out = torch.stack([visitation_dist, goal_inside_partial_dist], dim=1)
            return visitation_prob_or_log_prob_out, goal_outside_prob_or_logprob

        else:
            goal_dist = softmax_func(goal_inside_dist_scores.view(batch_size, width * height)).view(
                goal_inside_dist_scores.size())
            # Re-assemble back into the Bx2xHxW tensor representation
            visitation_prob_or_log_prob_out = torch.stack([visitation_dist, goal_dist], dim=1)
            return visitation_prob_or_log_prob_out