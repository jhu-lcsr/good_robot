import torch
import torch.nn.functional as F

from learning.modules.module_base import ModuleBase


class MultiscaleStack(ModuleBase):

    def __init__(self, scales, output_size_xy):
        super(MultiscaleStack, self).__init__()
        self.output_size = output_size_xy
        self.is_cuda = False
        self.cuda_device = None
        self.scales = scales
        # e.g. scales = [1.0, 0.5, 0.25]

    def cuda(self, device=None):
        self.is_cuda = True
        self.cuda_device = device

    def init_weights(self):
        pass

    def crop_output(self, output_map):
        out_x = self.output_size[0]
        out_y = self.output_size[1]
        in_x = output_map.size(2)
        in_y = output_map.size(3)
        l_gap_x = int((in_x - out_x) / 2)
        u_gap_y = int((in_y - out_y) / 2)
        return output_map[:, :, l_gap_x:l_gap_x+out_x, u_gap_y:u_gap_y+out_y]

    def forward_scale(self, input, scale):
        if scale == 1:
            return self.crop_output(input)
        elif 0 < scale < 1:
            stride = int(1/scale)
            rescaled = F.avg_pool2d(input, stride)
            return self.crop_output(rescaled)
        else:
            print("Invalid scale! Must be int in range (0; 1] ")

    def forward(self, image):
        catlist = []
        for i, scale in enumerate(self.scales):
            catlist.append(self.forward_scale(image, scale))

        return torch.cat(catlist, dim=1)