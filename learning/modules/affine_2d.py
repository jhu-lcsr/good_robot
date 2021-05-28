import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from learning.inputs.common import np_to_tensor
from utils.simple_profiler import SimpleProfiler

PROFILE = False


class Affine2D(nn.Module):

    def __init__(self):
        super(Affine2D, self).__init__()
        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

    def cuda(self, device=None):
        nn.Module.cuda(self, device)
        return self

    def get_pytorch_to_img_mat(self, img_size, inv=False):
        """
        Returns an affine transformation matrix that takes an image in coordinate range [-1,1] and turns it
        into an image of coordinate range [W,H]
        :param img_size: (W,H)
        :return:
        """
        # First move the image so that the origin is in the top-left corner
        # (in pytorch, the origin is in the center of the image)
        """
        t1 = np.asarray([
            [1.0, 0, 1.0],
            [0, 1.0, 1.0],
            [0, 0, 1.0]
        ])

        # Then scale the image up to the required size
        scale_w = img_size[0] / 2
        scale_h = img_size[1] / 2
        t2 = np.asarray([
            [scale_h, 0, 0],
            [0, scale_w, 0],
            [0, 0, 1]
        ])
        """

        # First scale the image to pixel coordinates
        scale_w = img_size[0] / 2
        scale_h = img_size[1] / 2

        t1 = np.asarray([
            [scale_h, 0, 0],
            [0, scale_w, 0],
            [0, 0, 1]
        ])

        # Then move it such that the corner is at the origin
        t2 = np.asarray([
            [1.0, 0, scale_h],
            [0, 1.0, scale_w],
            [0, 0, 1.0]
        ])

        T = np.dot(t2, t1)

        if inv:
            T = np.linalg.inv(T)

        T_t = np_to_tensor(T, cuda=False)

        return T_t

    def img_affines_to_pytorch_cpu(self, img_affines, img_in_size, out_size):
        T_src = self.get_pytorch_to_img_mat(img_in_size, inv=False)
        Tinv_dst = self.get_pytorch_to_img_mat(out_size, inv=True)

        self.prof.tick("getmat")

        # Convert pytorch-coord image to imgage pixel coords, apply the transformation, then convert the result back.
        batch_size = img_affines.size(0)
        T_src = T_src.repeat(batch_size, 1, 1)
        Tinv_dst = Tinv_dst.repeat(batch_size, 1, 1)

        x = torch.bmm(img_affines, T_src)       # Convert pytorch coords to pixel coords and apply the transformation
        pyt_affines = torch.bmm(Tinv_dst, x)    # Convert the transformation back to pytorch coords

        self.prof.tick("convert")

        inverses = [torch.inverse(affine) for affine in pyt_affines]

        self.prof.tick("inverse")

        pyt_affines_inv = torch.stack(inverses, dim=0)

        self.prof.tick("stack")

        return pyt_affines_inv

    def forward(self, image, affine_mat, out_size=None):
        """
        Applies the given batch of affine transformation matrices to the batch of images
        :param image:   batch of images to transform
        :param affine:  batch of affine matrices to apply. Specified in image coordinates (internally converted to pytorch coords)
        :return:        batch of images of same size as the input batch with the affine matrix having been applied
        """

        batch_size = image.size(0)

        self.prof.tick(".")

        # Cut off the batch and channel to get the image size as the source size
        img_size = list(image.size())[2:4]
        if out_size is None:
            out_size = img_size

        affines_pytorch = self.img_affines_to_pytorch_cpu(affine_mat, img_size, out_size)
        affines_pytorch = affines_pytorch.to(image.device)

        # Build the affine grid
        grid = F.affine_grid(affines_pytorch[:, [0,1], :], torch.Size((batch_size, 1, out_size[0], out_size[1]))).float()

        self.prof.tick("affine_grid")

        # Rotate the input image
        rot_img = F.grid_sample(image, grid, padding_mode="zeros")

        self.prof.tick("grid_sample")
        self.prof.loop()
        self.prof.print_stats(10)

        return rot_img