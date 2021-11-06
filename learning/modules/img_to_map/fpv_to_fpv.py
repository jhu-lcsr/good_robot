import torch
from torch import nn as nn
import torch.nn.functional as F

from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.img_to_img.img_to_features import ImgToFeatures

from visualization import Presenter
from utils.simple_profiler import SimpleProfiler

PROFILE = False


class FPVToFPVMap(nn.Module):
    def __init__(self, img_w, img_h, res_channels, map_channels, img_dbg=False):
        super(FPVToFPVMap, self).__init__()

        self.image_debug = img_dbg

        # Provide enough padding so that the map is scaled down by powers of 2.
        self.img_to_features = ImgToFeatures(res_channels, map_channels, img_w, img_h)
        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.actual_images = None

    def init_weights(self):
        self.img_to_features.init_weights()

    def reset(self):
        self.actual_images = None

    def forward_fpv_features(self, images, sentence_embeds, parent=None):
        """
        Compute the first-person image features given the first-person images
        If grounding loss is enabled, will also return sentence_embedding conditioned image features
        :param images: images to compute features on
        :param sentence_embeds: sentence embeddings for each image
        :param parent:
        :return: features_fpv_vis - the visual features extracted using the ResNet
                 features_fpv_gnd - the grounded visual features obtained after applying a 1x1 language-conditioned conv
        """
        # Extract image features. If they've been precomputed ahead of time, just grab it by the provided index
        features_fpv_vis = self.img_to_features(images)
        return features_fpv_vis

    def forward(self, images, poses, sentence_embeds, parent=None, show=""):

        self.prof.tick("out")

        features_fpv_vis_only = self.forward_fpv_features(images, sentence_embeds, parent)

        return features_fpv_vis_only