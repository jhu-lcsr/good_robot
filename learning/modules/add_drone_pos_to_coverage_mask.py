import torch
import torch.nn as nn
import numpy as np
import math

from transformations import pos_m_to_px

from visualization import Presenter

class AddDroneInitPosToCoverage(nn.Module):

    def __init__(self, world_size_px, world_size_m, map_size_px):
        super(AddDroneInitPosToCoverage, self).__init__()
        self.world_size_px = world_size_px
        self.world_size_m = world_size_m
        self.map_size_px = map_size_px

        self.hfov = 86 * math.pi / 180
        self.radius = 8
        self.full_radius = 2
        self.current_pos_mask = self._generate_mask()

    def cuda(self, device=None):
        nn.Module.cuda(self, device)
        self.current_pos_mask.cuda(device)

    def to(self, device):
        self.current_pos_mask.to(device)

    def _generate_mask(self):
        m = torch.zeros([self.map_size_px, self.map_size_px])
        c_x, c_y = [int(self.map_size_px / 2)] * 2
        c_x_a = c_x - 2
        c_y_a = c_y
        for x in range(c_x - self.radius, c_x + self.radius):
            for y in range(c_y - self.radius, c_y + self.radius):
                angle = math.atan2(y - c_y_a, x - c_x_a)
                dst = math.sqrt((y-c_y) ** 2 + (x - c_x) ** 2)
                if (-self.hfov / 2 < angle < self.hfov / 2 and dst < self.radius) or dst <= self.full_radius:
                    m[y, x] = 1.0

        if False:
            Presenter().show_image(m, "init_pos_mask", scale=4, waitkey=True)
        return m

    def get_init_pos_masks(self, batch_size, device):
        self.current_pos_mask = self.current_pos_mask.to(device)
        return self.current_pos_mask[np.newaxis, np.newaxis, :, :].repeat([batch_size, 1, 1, 1])

    def forward(self, coverage_masks, initpos_masks):
        batch_size = coverage_masks.shape[0]
        coverage_masks_initpos = (coverage_masks + initpos_masks).clamp(0,1)

        if False:
            for i in range(batch_size):
                Presenter().show_image(coverage_masks[i,0], "cov_mask_before", scale=4, waitkey=1)
                Presenter().show_image(coverage_masks_initpos[i, 0], "cov_mask_after", scale=4, waitkey=True)
        
        return coverage_masks_initpos
