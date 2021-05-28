import torch
import torch.nn as nn
from torch.autograd import Variable
from learning.inputs.common import empty_float_tensor
from learning.modules.unet.lingunet_5 import Lingunet5
from learning.modules.unet.lingunet_5_s import Lingunet5S
from learning.modules.unet.lingunet_5_oob import Lingunet5OOB
from learning.modules.unet.lingunet_5_dualhead import Lingunet5DualHead
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d
from copy import deepcopy

sentence_embedding_size = 120
sentence_embedding_layers = 1
word_embedding_size = 20


class RatioPathPredictor(nn.Module):

    def __init__(self, lingunet_params,
                 prior_channels_in,
                 posterior_channels_in,
                 dual_head=False,
                 compute_prior=True,
                 use_prior=False,
                 oob=False):
        super(RatioPathPredictor, self).__init__()

        self.use_prior = use_prior
        self.prior_img_channels = prior_channels_in
        self.posterior_img_channels = posterior_channels_in
        self.dual_head = dual_head
        self.small_network = lingunet_params.get("small_network")
        self.oob = oob

        if use_prior:
            assert compute_prior, "If we want to use the prior distribution, we should compute it, right?"

        if self.oob:
            lingunet_params["in_channels"] = posterior_channels_in
            self.unet_posterior = Lingunet5OOB(deepcopy(lingunet_params))
            lingunet_params["in_channels"] = prior_channels_in
            self.unet_prior = Lingunet5OOB(deepcopy(lingunet_params))
        else:
            lingunet_params["in_channels"] = posterior_channels_in
            self.unet_posterior = Lingunet5(deepcopy(lingunet_params))
            lingunet_params["in_channels"] = prior_channels_in
            self.unet_prior = Lingunet5(deepcopy(lingunet_params))

        self.softmax = SpatialSoftmax2d()
        self.norm = nn.InstanceNorm2d(2)
        self.compute_prior = compute_prior

        self.dbg_t = None

    def init_weights(self):
        self.unet_posterior.init_weights()
        self.unet_prior.init_weights()

    def forward(self, image, sentence_embeddimg, map_poses, tensor_store=None, show=""):

        # TODO: Move map perturb data augmentation in here.
        if image.size(1) > self.posterior_img_channels:
            image = image[:, 0:self.posterior_img_channels, :, :]

        # channel 0 is start position
        # channels 1-3 are the grounded map
        # all other channels are the semantic map
        fake_embedding = torch.zeros([image.size(0), 1]).to(image.device)

        # The first N channels would've been computed by grounding map processor first. Remove them so that the
        # prior is clean from any language

        posterior_distributions = self.unet_posterior(image, sentence_embeddimg, tensor_store)

        if self.compute_prior:
            lang_conditioned_channels = self.posterior_img_channels - self.prior_img_channels
            prior_image = image[:, lang_conditioned_channels:]
            prior_distributions = self.unet_prior(prior_image, fake_embedding)

        return posterior_distributions, map_poses
