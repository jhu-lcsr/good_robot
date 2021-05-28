import torch
from torch import nn as nn

from learning.inputs.common import empty_float_tensor
from learning.modules.rss.map_lang_semantic_filter import MapLangSemanticFilter
from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.img_to_img.img_to_features import ImgToFeatures
from learning.models.semantic_map.grid_sampler import GridSampler

from visualization import Presenter
from utils.simple_profiler import SimpleProfiler

PROFILE = False


class FPVToEgoMap(MapTransformerBase):
    # TODO: This is broken now. Must project to global map, then transform to ego map
    def __init__(self,
                 source_map_size, world_size_px,
                 world_size_m, img_w, img_h,
                 embed_size, map_channels, gnd_channels, res_channels=32,
                 lang_filter=False, img_dbg=False):
        super(FPVToEgoMap, self).__init__(source_map_size, world_size_px, world_size_m=world_size_m)

        self.image_debug = img_dbg
        self.use_lang_filter = lang_filter

        # Process images using a resnet to get a feature map
        if self.image_debug:
            self.img_to_features = nn.MaxPool2d(8)
        else:
            # Provide enough padding so that the map is scaled down by powers of 2.
            self.img_to_features = ImgToFeatures(res_channels, map_channels)

        if self.use_lang_filter:
            self.lang_filter = MapLangSemanticFilter(embed_size, map_channels, gnd_channels)

        # Project feature maps to the global frame
        self.map_projection = PinholeCameraProjectionModule(
            source_map_size, world_size_px, world_size_px, source_map_size / 2, img_w, img_h)

        self.grid_sampler = GridSampler()

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

        self.actual_images = None

    def cuda(self, device=None):
        MapTransformerBase.cuda(self, device)
        self.map_projection.cuda(device)
        self.grid_sampler.cuda(device)
        self.img_to_features.cuda(device)
        if self.use_lang_filter:
            self.lang_filter.cuda(device)

    def init_weights(self):
        if not self.image_debug:
            self.img_to_features.init_weights()

    def reset(self):
        self.actual_images = None
        super(FPVToEgoMap, self).reset()

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

        if parent is not None:
            parent.keep_inputs("fpv_features", features_fpv_vis)
        self.prof.tick("feat")

        # If required, pre-process image features by grounding them in language
        if self.use_lang_filter:
            self.lang_filter.precompute_conv_weights(sentence_embeds)
            features_gnd = self.lang_filter(features_fpv_vis)
            if parent is not None:
                parent.keep_inputs("fpv_features_g", features_gnd)
            self.prof.tick("gnd")
            return features_fpv_vis, features_gnd

        return features_fpv_vis, None

    def forward(self, images, poses, sentence_embeds, parent=None, show=""):

        self.prof.tick("out")

        features_fpv_vis_only, features_fpv_gnd_only = self.forward_fpv_features(images, sentence_embeds, parent)

        # If we have grounding features, the overall features are a concatenation of grounded and non-grounded features
        if features_fpv_gnd_only is not None:
            features_fpv_all = torch.cat([features_fpv_gnd_only, features_fpv_vis_only], dim=1)
        else:
            features_fpv_all = features_fpv_vis_only

        # Project first-person view features on to the map in egocentric frame
        grid_maps = self.map_projection(poses)
        self.prof.tick("proj_map")
        features_r = self.grid_sampler(features_fpv_all, grid_maps)

        # Obtain an ego-centric map mask of where we have new information
        ones_size = list(features_fpv_all.size())
        ones_size[1] = 1
        tmp_ones = empty_float_tensor(ones_size, self.is_cuda, self.cuda_device).fill_(1.0)
        new_coverages = self.grid_sampler(tmp_ones, grid_maps)

        # Make sure that new_coverage is a 0/1 mask (grid_sampler applies bilinear interpolation)
        new_coverages = new_coverages - torch.min(new_coverages)
        new_coverages = new_coverages / torch.max(new_coverages)

        self.prof.tick("gsample")

        if show != "":
            Presenter().show_image(images.data[0, 0:3], show + "_img", torch=True, scale=1, waitkey=1)
            Presenter().show_image(features_r.data[0, 0:3], show, torch=True, scale=6, waitkey=1)
            Presenter().show_image(new_coverages.data[0], show + "_covg", torch=True, scale=6, waitkey=1)

        self.prof.loop()
        self.prof.print_stats(10)

        return features_r, new_coverages