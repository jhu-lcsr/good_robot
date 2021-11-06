import torch
from torch import nn as nn

from learning.models.semantic_map.map_affine import MapAffine
from learning.modules.map_transformer_base import MapTransformerBase

from visualization import Presenter


class TopDownToEgoImg(MapTransformerBase):
    def __init__(self, img_in_size=256, world_size_px=32):
        super(TopDownToEgoImg, self).__init__(img_in_size, world_size_px, world_size_m=world_size_px)
        self.is_cuda = False
        self.cuda_device = None

        # Process images using a resnet to get a feature map
        self.feature_net = nn.AvgPool2d(8, stride=8)

        self.map_affine = MapAffine(source_map_size=img_in_size, world_size_px=world_size_px)

    def cuda(self, device=None):
        MapTransformerBase.cuda(self, device)
        self.is_cuda = True
        self.cuda_device = device
        self.map_affine.cuda(device)
        return self

    def init_weights(self):
        pass

    def forward(self, image_g, pose):

        self.set_map(image_g, None)
        image_r, _ = self.get_map(pose)

        presenter = Presenter()
        presenter.show_image(image_g[0].data, "img_g", torch=True, waitkey=False, scale=2)
        presenter.show_image(image_r[0].data, "img_r", torch=True, waitkey=100, scale=2)

        features_r = self.feature_net(image_r)
        coverage = torch.ones_like(features_r)
        return features_r, coverage, image_r