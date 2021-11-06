import torch.nn as nn
from torch.autograd import Variable
from learning.inputs.common import empty_float_tensor
from learning.modules.unet.lingunet_5 import Lingunet5
from learning.modules.unet.lingunet_5_s import Lingunet5S
from learning.modules.unet.lingunet_5_oob import Lingunet5OOB
from learning.modules.unet.lingunet_5_dualhead import Lingunet5DualHead
from learning.modules.spatial_softmax_2d import SpatialSoftmax2d
from copy import deepcopy

from learning.modules.cuda_module import CudaModule

sentence_embedding_size = 120
sentence_embedding_layers = 1
word_embedding_size = 20


class RatioPathPredictor(CudaModule):

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
        elif self.small_network:
            lingunet_params["in_channels"] = posterior_channels_in
            self.unet_posterior = Lingunet5S(deepcopy(lingunet_params))
            lingunet_params["in_channels"] = prior_channels_in
            self.unet_prior = Lingunet5S(deepcopy(lingunet_params))
        elif self.dual_head:
            lingunet_params["in_channels"] = posterior_channels_in
            self.unet_posterior = Lingunet5DualHead(deepcopy(lingunet_params))
            lingunet_params["in_channels"] = prior_channels_in
            self.unet_prior = Lingunet5DualHead(deepcopy(lingunet_params))
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

    def cuda(self, device=None):
        CudaModule.cuda(self, device)
        #self.map_filter.cuda(device)
        self.softmax.cuda(device)
        self.dbg_t = None
        return self

    def forward(self, image, sentence_embeddimg, map_poses, tensor_store=None, show=""):

        # TODO: Move map perturb data augmentation in here.
        if image.size(1) > self.posterior_img_channels:
            image = image[:, 0:self.posterior_img_channels, :, :]

        # channel 0 is start position
        # channels 1-3 are the grounded map
        # all other channels are the semantic map
        fake_embedding = Variable(empty_float_tensor([image.size(0), 1], self.is_cuda, self.cuda_device))

        # The first N channels would've been computed by grounding map processor first. Remove them so that the
        # prior is clean from any language

        tmp1 = self.unet_posterior(image, sentence_embeddimg, tensor_store)
        if self.dual_head:
            pred_mask_posterior, second_output_posterior = tmp1
        elif self.oob:
            pred_mask_posterior, goal_oob_score = tmp1
        else:
            pred_mask_posterior = tmp1

        #pred_mask_posterior_prob = self.softmax(pred_mask_posterior)

        if self.compute_prior:
            lang_conditioned_channels = self.posterior_img_channels - self.prior_img_channels
            prior_image = image[:, lang_conditioned_channels:]
            tmp2 = self.unet_prior(prior_image, fake_embedding)
            if self.dual_head:
                pred_mask_prior, second_output_prior = tmp2
            elif self.oob:
                pred_mask_prior, goal_oob_prior_score = tmp2
            else:
                pred_mask_prior = tmp2
            #pred_mask_prior_prob = self.softmax(pred_mask_prior)
            #ratio_mask = pred_mask_posterior_prob / (pred_mask_prior_prob + 1e-3)
            #ratio_mask = self.softmax(ratio_mask)
        #else:
        #    pred_mask_prior_prob = pred_mask_posterior

        #if show != "":
        #    Presenter().show_image(ratio_mask.data[i], show, torch=True, scale=8, waitkey=1)

        #self.set_maps(pred_mask_posterior_prob, map_poses)

        # TODO: Careful whether we use a prob dist or scores
        ret = pred_mask_posterior
        #if self.use_prior:
        #    ret = pred_mask_prior_prob

        if self.dual_head:
            return ret, map_poses, second_output_posterior
        elif self.oob:
            return ret, goal_oob_score, map_poses
        else:
            return ret, map_poses