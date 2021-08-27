import torch
import numpy as np
import math

from transformations import pos_m_to_px

from learning.modules.cuda_module import CudaModule
from visualization import Presenter

class AddDroneInitPosToCoverage(CudaModule):

    def __init__(self, world_size_px, world_size_m, map_size_px):
        super(AddDroneInitPosToCoverage, self).__init__()
        self.world_size_px = world_size_px
        self.world_size_m = world_size_m
        self.map_size_px = map_size_px

        self.hfov = 84 * math.pi / 180
        self.radius = 5.0
        self.current_pos_mask = self._generate_mask()


    def cuda(self, device=None):
        CudaModule.cuda(self, device)

    def _generate_mask(self):
        m = torch.zeros([self.map_size_px, self.map_size_px])
        c_x, c_y = self.map_size_px / 2
        for x in range(c_x - self.radius, c_x + self.radius):
            for y in range(c_y - self.radius, c_y + self.radius):
                dx = x - c_x
                dy = y - c_y
                angle = math.atan2(dy, dx)
                dst = math.sqrt(dy ** 2 + dx ** 2)
                if - self.hfov / 2 < angle < self.hfov / 2 and dst < self.radius:
                    m[c_x, c_y] = 1.0

        if False:
            Presenter().show_image(m, "init_pos_mask", scale=4, waitkey=True)

    def forward(self, coverage_masks_w, cam_poses):
        pos_px = pos_m_to_px(cam_poses.position[0:1], img_size_px=self.world_size_px, world_size_px=self.world_size_px, world_size_m=self.world_size_m)
        batch_size = coverage_masks_w.shape[0]
        # TODO: Don't do this at test-time for everything except the first action!
        assert cam_poses.position.shape[0] > 0, "Not implemented test-time behavior"
        pos_mask = torch.zeros_like(coverage_masks_w[0,0])
        radius = 6 # 6 pixels is a bit less than a meter

        x = pos_px[0][0].item()
        y = pos_px[0][1].item()

        xi = int(x)
        yi = int(y)
        min_x = max(xi-radius, 0)
        min_y = max(yi-radius, 0)
        max_x = min(xi+radius, coverage_masks_w.shape[2])
        max_y = min(yi+radius, coverage_masks_w.shape[2])

        indices = [[i,j] for i in range(min_x, max_x) for j in range(min_y, max_y) if (x-i-0.5)**2 + (y-j-0.5)**2 < radius ** 2]
        for i,j in indices:
            pos_mask[i,j] = 1.0

        coverage_masks_w_init_pos = (coverage_masks_w + pos_mask[np.newaxis, np.newaxis, :, :]).clamp(0,1)

        if True:
            for i in range(batch_size):
                Presenter().show_image(coverage_masks_w[i,0], "cov_mask_before", scale=4, waitkey=1)
                Presenter().show_image(coverage_masks_w_init_pos[i, 0], "cov_mask_after", scale=4, waitkey=True)
        
        return coverage_masks_w_init_pos
