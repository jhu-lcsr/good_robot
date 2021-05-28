import torch
import torch.nn as nn

from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.rss.map_lang_semantic_filter import MapLangSemanticFilter
from learning.modules.rss.map_lang_spatial_filter import MapLangSpatialFilter
from visualization import Presenter


class LangFilterMapProcessor(nn.Module):

    def __init__(self, embed_size, in_channels, out_channels, spatial=False, cat_out=False):
        super(LangFilterMapProcessor, self).__init__()
        self.embed_size = embed_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cat_out = cat_out

        if spatial:
            self.lang_filter = MapLangSpatialFilter(embed_size, in_channels, out_channels)
        else:
            self.lang_filter = MapLangSemanticFilter(embed_size, in_channels, out_channels)

    def init_weights(self):
        self.lang_filter.init_weights()

    def forward(self, images, sentence_embeddings, map_poses, proc_mask=None, show=""):

        # If we are supposed to use less channels than the input map has, just grab the first N channels
        if images.size(1) > self.in_channels:
            images_in = images[:, 0:self.in_channels, :, :]
        else:
            images_in = images

        # Apply the language-conditioned convolutional filter
        self.lang_filter.precompute_conv_weights(sentence_embeddings)
        images_out = self.lang_filter(images_in)

        if show != "":
            Presenter().show_image(images_out.data[0, 0:3], show, torch=True, scale=4, waitkey=1)

        # If requested, concatenate with the prior input, such that the first feature maps are from output
        # That allows chaining these modules and slicing
        if self.cat_out:
            images_out = torch.cat([images_out, images_in], dim=1)

        return images_out, map_poses