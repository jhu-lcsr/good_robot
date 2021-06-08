import torch
from torch import nn as nn
import torch.nn.functional as F

from learning.inputs.common import empty_float_tensor
from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.img_to_img.img_to_features import ImgToFeatures
from learning.models.semantic_map.grid_sampler import GridSampler
from learning.models.semantic_map.pinhole_camera_inv import PinholeCameraProjectionModuleGlobal

from visualization import Presenter
from utils.simple_profiler import SimpleProfiler

from learning.inputs.pose import Pose

PROFILE = False
# Set to true to project first-person images instead of feature maps.
DEBUG_WITH_IMG = True

class FPVToGlobalMap(MapTransformerBase):
    def __init__(self,
                 source_map_size,
                 world_size_px,
                 world_size_m,
                 img_w,
                 img_h,
                 res_channels,
                 map_channels,
                 cam_h_fov,
                 domain,
                 img_dbg=False):

        super(FPVToGlobalMap, self).__init__(source_map_size, world_size_px, world_size_m)

        self.image_debug = img_dbg

        self.use_lang_filter = False

        # Process images using a resnet to get a feature map
        if self.image_debug:
            self.img_to_features = nn.MaxPool2d(8)
        else:
            # Provide enough padding so that the map is scaled down by powers of 2.
            self.img_to_features = ImgToFeatures(res_channels, map_channels, img_w, img_h)

        # Project feature maps to the global frame
        self.map_projection = PinholeCameraProjectionModuleGlobal(
            source_map_size, world_size_px, world_size_m, img_w, img_h, cam_h_fov, domain)

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
        super(FPVToGlobalMap, self).reset()

    def forward_fpv_features(self, images, sentence_embeds, tensor_store=None):
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

        if tensor_store is not None:
            tensor_store.keep_inputs("fpv_features", features_fpv_vis)
        #self.prof.tick("feat")

        # If required, pre-process image features by grounding them in language
        if self.use_lang_filter:
            self.lang_filter.precompute_conv_weights(sentence_embeds)
            features_gnd = self.lang_filter(features_fpv_vis)
            if tensor_store is not None:
                tensor_store.keep_inputs("fpv_features_g", features_gnd)
            #self.prof.tick("gnd")
            return features_fpv_vis, features_gnd

        return features_fpv_vis, None

    def forward(self, images, poses, sentence_embeds, tensor_store=None, show="", halfway=False):

        self.prof.tick("out")

        # self.map_projection is implemented in numpy on CPU.
        # If we give it poses on the GPU, it will transfer them to the CPU, which causes a CUDA SYNC and waits for the
        # ResNet forward pass to complete. To make use of full GPU/CPU concurrency, we move the poses to the cpu first
        poses_cpu = poses.cpu()

        features_fpv_vis_only, features_fpv_gnd_only = self.forward_fpv_features(images, sentence_embeds, tensor_store)

        # Halfway HAS to be True and not only truthy
        if halfway == True:
            return None, None

        # If we have grounding features, the overall features are a concatenation of grounded and non-grounded features
        if features_fpv_gnd_only is not None:
            features_fpv_all = torch.cat([features_fpv_gnd_only, features_fpv_vis_only], dim=1)
        else:
            features_fpv_all = features_fpv_vis_only

        # Project first-person view features on to the map in egocentric frame
        grid_maps_cpu = self.map_projection(poses_cpu)
        grid_maps = grid_maps_cpu.to(features_fpv_all.device)

        self.prof.tick("proj_map_and_features")
        features_r = self.grid_sampler(features_fpv_all, grid_maps)

        if DEBUG_WITH_IMG:
            img_w = self.grid_sampler(images, grid_maps)
            if tensor_store is not None:
                tensor_store.keep_inputs("images_w", img_w)
            #Presenter().show_image(images.data[0], "fpv_raw", torch=True, scale=2, waitkey=1)
            #Presenter().show_image(img_w.data[0], "fpv_projected", torch=True, scale=2, waitkey=1)

        # Obtain an ego-centric map mask of where we have new information
        ones_size = list(features_fpv_all.size())
        ones_size[1] = 1
        tmp_ones = torch.ones(ones_size).to(features_r.device)
        new_coverages = self.grid_sampler(tmp_ones, grid_maps)

        # Make sure that new_coverage is a 0/1 mask (grid_sampler applies bilinear interpolation)
        new_coverages = new_coverages - torch.min(new_coverages)
        new_coverages = new_coverages / (torch.max(new_coverages) + 1e-18)

        self.prof.tick("gsample")

        if show != "":
            Presenter().show_image(images.data[0, 0:3], show + "fpv_img", torch=True, scale=2, waitkey=1)

            grid_maps_np = grid_maps.data[0].numpy()

            Presenter().show_image(grid_maps_np, show + "_grid", torch=False, scale=4, waitkey=1)
            Presenter().show_image(features_fpv_all.data[0, 0:3], show + "_preproj", torch=True, scale=8, waitkey=1)
            Presenter().show_image(images.data[0, 0:3], show + "_img", torch=True, scale=1, waitkey=1)
            Presenter().show_image(features_r.data[0, 0:3], show + "_projected", torch=True, scale=6, waitkey=1)
            Presenter().show_image(new_coverages.data[0], show + "_covg", torch=True, scale=6, waitkey=1)

        self.prof.loop()
        self.prof.print_stats(10)

        return features_r, new_coverages