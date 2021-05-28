import torch
from data_io.weights import enable_weight_saving

from learning.modules.resnet.resnet_13_light import ResNet13Light
from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.rss.map_lang_semantic_filter import MapLangSemanticFilter

from visualization import Presenter


class TopDownToEgoMap(MapTransformerBase):
    def __init__(self, img_in_size=256, world_size_in_img=256, feature_channels=32, ground_channels=3, embed_size=40, aux_ground=False, freeze=False):
        super(TopDownToEgoMap, self).__init__(img_in_size, world_size_in_img)

        # Process images using a resnet to get a feature map
        self.feature_net = ResNet13Light(feature_channels, down_pad=True)

        self.aux_ground = aux_ground
        if aux_ground:
            self.lang_filter = MapLangSemanticFilter(embed_size, feature_channels, ground_channels)
            enable_weight_saving(self.lang_filter, "ground_filter", alwaysfreeze=freeze)

        enable_weight_saving(self.feature_net, "feature_resnet_light", alwaysfreeze=freeze)

    def cuda(self, device=None):
        MapTransformerBase.cuda(self, device)
        self.map_affine.cuda(device)
        if self.aux_ground:
            self.lang_filter.cuda(device)
        return self

    def init_weights(self):
        self.feature_net.init_weights()

    def forward(self, image_g, pose, sentence_embed, parent=None, show=""):

        # scale to 0-1 range
        #image_g = image_g - torch.min(image_g)
        #image_g = image_g / (torch.max(image_g) + 1e-9)

        # rotate to robot frame
        # TODO: Temporarily changed to local pose
        self.set_map(image_g, pose)
        image_r, _ = self.get_map(pose)


        """
        # normalize mean-0 std-1
        image_r = image_r - torch.mean(image_r)
        image_r = image_r / (torch.std(image_r) + 1e-9)

        ones = torch.ones_like(image_g)
        self.set_map(ones, None)
        cov_r, _ = self.get_map(pose)
        cov_r = cov_r - torch.min(cov_r)
        cov_r /= (torch.max(cov_r) + 1e-9)
        cov_rl = cov_r > 1e-8

        blackcolor = torch.min(image_g)

        #image_r[cov_rl] = blackcolor
        """

        features_r = self.feature_net(image_r)

        if parent is not None:
            parent.keep_inputs("fpv_features", features_r)

        if self.aux_ground:
            self.lang_filter.precompute_conv_weights(sentence_embed)
            features_g = self.lang_filter(features_r)
            if parent is not None:
                parent.keep_inputs("fpv_features_g", features_g)

            features_all = torch.cat([features_g, features_r], dim=1)
        else:
            features_all = features_r

        coverage = torch.ones_like(features_all)

        if show != "":
            Presenter().show_image(image_r.data[0, 0:3], show + "_img", torch=True, scale=1, waitkey=20)
            Presenter().show_image(features_r.data[0, 0:3], show, torch=True, scale=12, waitkey=20)
            #Presenter().show_image(cov_r.data[0, 0:3], show+ "_convg", torch=True, scale=1, waitkey=20)

        return features_all, coverage