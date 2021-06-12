import torch
from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.unet.unet_5_contextual_bneck3 import Unet5ContextualBneck
from learning.modules.unet.lingunet_5 import Lingunet5
from learning.modules.unet.lingunet_5_s import Lingunet5S
from learning.inputs.partial_2d_distribution import Partial2DDistribution

from learning.modules.dbg_writer import DebugWriter

sentence_embedding_size = 120
sentence_embedding_layers = 1
word_embedding_size = 20


# TODO: Rename to ImgToImg
# TODO: Remove the transforming bits - that should be handled by MapTransformer
class PathPredictor(MapTransformerBase):

    # TODO: Standardize run_params
    def __init__(self, lingunet_params, source_map_size, world_size_px, world_size_m):
        super(PathPredictor, self).__init__(source_map_size, world_size_px, world_size_m)

        if lingunet_params.get("small_network"):
            self.unet = Lingunet5S(lingunet_params)
        else:
            self.unet = Lingunet5(lingunet_params)

        #self.map_filter = MapLangSemanticFilter(emb_size, feature_channels, 3)
        self.map_size_px = source_map_size
        self.world_size_px = world_size_px
        self.world_size_m = world_size_m

        self.dbg_t = None
        self.seq = 0

    def init_weights(self):
        self.unet.init_weights()

    def reset(self):
        super(PathPredictor, self).reset()
        self.seq = 0

    def cuda(self, device=None):
        MapTransformerBase.cuda(self, device)
        #self.map_filter.cuda(device)
        self.dbg_t = None
        return self

    def dbg_write_extra(self, map, pose):
        if DebugWriter().should_write():
            self.seq += 1
            # Initialize a transformer module
            if self.dbg_t is None:
                self.dbg_t = MapTransformerBase(self.map_size_px, self.world_size_px, self.world_size_m).to(map.device)

            # Transform the prediction to the global frame and write out to disk.
            self.dbg_t.set_map(map, pose)
            map_global, _ = self.dbg_t.get_map(None)
            DebugWriter().write_img(map_global[0], "gif_overlaid", args={"world_size": self.world_size_px, "name": "pathpred"})

    def forward(self, image, sentence_embeddimg, map_poses, proc_mask=None, show=""):
        # TODO: Move map perturb data augmentation in here.

        if image.size(1) > self.feature_channels:
            image = image[:, 0:self.feature_channels, :, :]

        pred_mask = self.unet(image, sentence_embeddimg)

        # Wtf is this:
        #self.map_filter.precompute_conv_weights(sentence_embeddimg)
        #features_filtered = self.map_filter(image)
        #out_maps = torch.cat([pred_mask, features_filtered], dim=1)

        """
        if proc_mask is not None:
            bs = pred_mask.size(0)
            for i in range(bs):

                # If we are using this processed map, apply it
                if proc_mask[bs]:
                    self.set_map(pred_mask[i:i+1], map_poses[i:i+1])

                # Otherwise return the latest processed map, rotated in this frame of reference
                pred_mask[i] = self.get_map(map_poses[i:i+1])

        if show != "":
            Presenter().show_image(pred_mask.data[0], show, torch=True, scale=8, waitkey=1)

        self.set_maps(pred_mask, map_poses)

        #self.dbg_write_extra(pred_mask, map_poses)
        """
        return pred_mask, map_poses
