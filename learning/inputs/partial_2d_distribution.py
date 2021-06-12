import numpy as np

import torch
import torch.nn as nn

from visualization import Presenter


class Partial2DDistribution(torch.nn.Module):

    def __init__(self, inner_distribution, outer_prob_mass):
        super(Partial2DDistribution, self).__init__()
        self.inner_distribution = inner_distribution
        self.outer_prob_mass = outer_prob_mass
        self.log_softmax_module = nn.LogSoftmax(dim=2)
        self.softmax_module = nn.Softmax(dim=2)

    @classmethod
    def from_distribution_and_mask(cls, v_dist, cov_mask):
        # Masks a visitation distribution and creates a Partial2DDistribution
        batch_size = v_dist.shape[0]
        channels = v_dist.shape[1]
        # Normalize before masking
        for c in range(channels):
            v_dist[:, c] /= (v_dist[:, c].view([batch_size, -1]).sum(dim=1)[:, np.newaxis, np.newaxis] + 1e-10)

        # Mask distribution
        v_dist_inner_masked = v_dist * cov_mask[:, np.newaxis, :, :]

        probs_inside = v_dist_inner_masked.view([batch_size, channels, -1]).sum(2)
        probs_outside = 1 - probs_inside
        v_dist_masked = Partial2DDistribution(v_dist_inner_masked, probs_outside)
        return v_dist_masked

    def to(self, *args, **kwargs):
        self.inner_distribution.to(*args, **kwargs)
        self.outer_prob_mass.to(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        self.inner_distribution.cuda(*args, **kwargs)
        self.outer_prob_mass.cuda(*args, **kwargs)
        return self

    def get_full_flat_distribution(self):
        batch_size = self.inner_distribution.shape[0]
        num_distributions = self.inner_distribution.shape[1]
        inner_flat = self.inner_distribution.view([batch_size, num_distributions, -1])
        outer = self.outer_prob_mass.view([batch_size, num_distributions, -1])
        full_flat = torch.cat([inner_flat, outer], dim=2)
        return full_flat

    def __index__(self, *args, **kwargs):
        new_inner = self.inner_distribution.__index__(*args, **kwargs)
        new_outer = self.outer_prob_mass.__index__(*args, **kwargs)
        return Partial2DDistribution(new_inner, new_outer)

    def detach(self):
        return Partial2DDistribution(self.inner_distribution.detach(), self.outer_prob_mass.detach())

    def softmax(self, logsoftmax=False):
        batch_size = self.inner_distribution.size(0)
        num_channels = self.inner_distribution.size(1)
        assert num_channels == 2, "Must have 2 channels: visitation distribution scores and goal distribution scores"
        height = self.inner_distribution.size(2)
        width = self.inner_distribution.size(3)

        flat_inner = self.inner_distribution.view([batch_size, num_channels, -1])
        flat_outer = self.outer_prob_mass.view([batch_size, num_channels, -1])
        flat_full = torch.cat([flat_inner, flat_outer], dim=2)

        softmax_func = self.log_softmax_module if logsoftmax else self.softmax_module

        flat_softmaxed = softmax_func(flat_full)

        new_inner_distribution = flat_softmaxed[:, :, :-1].view([batch_size, num_channels, height, width])
        new_outer_prob_mass = flat_softmaxed[:, :, -1]

        return Partial2DDistribution(new_inner_distribution, new_outer_prob_mass)

    def clone(self):
        return Partial2DDistribution(self.inner_distribution.clone(), self.outer_prob_mass.clone())

    def visualize(self, idx=0):
        width = self.inner_distribution.shape[3]
        height = self.inner_distribution.shape[2]
        channels = self.inner_distribution.shape[1]
        barwidth = int(width / 5)

        # Include 2 bars - stop and visitation
        showwidth = width + channels * barwidth
        showheight = height
        show_img = np.zeros((showheight, showwidth, channels))

        npinner = self.inner_distribution[idx].detach().cpu().numpy().transpose((1, 2, 0))
        for c in range(channels):
            npinner[:, :, c] /= (npinner[:, :, c].max() + 1e-10)

        show_img[0:height, 0:width, :] = npinner
        for c in range(channels):
            value = self.outer_prob_mass.detach()[idx,c].item()
            barheight = int(value * showheight)
            show_img[showheight - barheight:, width + c * barwidth:width + (c + 1) * barwidth, c] = 1.0
        return show_img

    def show(self, name, scale=8, waitkey=1, idx=0):
        show_img = self.visualize(idx)
        Presenter().show_image(show_img, name, scale=scale, waitkey=waitkey)
