#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time
try:
    import efficientnet_pytorch
    from efficientnet_pytorch import EfficientNet
except ImportError:
    print('efficientnet_pytorch is not available, using densenet. '
          'EfficientNets can be installed with the command:'
          '    pip3 install efficientnet-pytorch --user --upgrade'
          'See https://github.com/lukemelas/EfficientNet-PyTorch for details')
    efficientnet_pytorch = None


def tile_vector_as_image_channels_torch(vector_op, image_shape):
    """
    Takes a vector of length n and an image shape BCHW,
    and repeat the vector as channels at each pixel.

    Code source: https://github.com/ahundt/costar_dataset/blob/master/costar_dataset/block_stacking_reader_torch.py

    # Params
      vector_op: A tensor vector to tile.
      image_shape: A list of integers [width, height] with the desired dimensions.
    """
    # input vector shape
    ivs = vector_op.shape
    # print('image_shape: ' + str(image_shape))

    # reshape the vector into a single pixel
    vector_op = vector_op.reshape([ivs[0], ivs[1], 1, 1])
    # print('vector_op pre-repeat shape:' + str(vector_op.shape))

    # repeat the vector at every pixel according to the specified image shape
    vector_op = vector_op.expand([ivs[0], ivs[1], image_shape[2], image_shape[3]])
    # print('vector_op post-repeat shape:' + str(vector_op.shape))
    # print('vector_op first channel: ' + str(vector_op[0,:,0,0]))
    return vector_op


def trunk_net(name='', fc_channels=2048, goal_condition_len=0, channels_out=3):
    first_fc = fc_channels + goal_condition_len
    # original behavior of second conv layer
    # second_fc = 64
    # new behavior of second conv layer
    second_fc = fc_channels + goal_condition_len
    return nn.Sequential(OrderedDict([
            (name + '-norm0', nn.BatchNorm2d(first_fc)),
            (name + '-relu0', nn.ReLU(inplace=True)),
            (name + '-conv0', nn.Conv2d(first_fc, second_fc, kernel_size=1, stride=1, bias=False)),
            (name + '-norm1', nn.BatchNorm2d(second_fc)),
            (name + '-relu1', nn.ReLU(inplace=True)),
            (name + '-conv1', nn.Conv2d(second_fc, channels_out, kernel_size=1, stride=1, bias=False))
            # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))


def vector_block(name='', channels_in=4, fc_channels=2048, channels_out=2048):
    return nn.Sequential(OrderedDict([
            (name + '-vectorblock-lin0', nn.Linear(channels_in, fc_channels, bias=False)),
            (name + '-vectorblock-norm0', nn.BatchNorm2d(fc_channels)),
            (name + '-vectorblock-relu0', nn.ReLU(inplace=True)),
            (name + '-vectorblock-lin1', nn.Linear(fc_channels, channels_out, bias=False))
            (name + '-vectorblock-norm1', nn.BatchNorm2d(channels_out)),
            (name + '-vectorblock-relu1', nn.ReLU(inplace=True)),
            # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))


def layers_forward(obj, rotate_theta, input_color_data, input_depth_data, goal_condition, tiled_goal_condition=None, requires_grad=True):
    """ Reduces the repetitive forward pass code across multiple model classes. See pixel_net forward() and responsive_net forward().
    """
    # Compute sample grid for rotation BEFORE neural network
    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
    affine_mat_before.shape = (2,3,1)
    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
    if obj.use_cuda:
        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=requires_grad).cuda(), input_color_data.size())
    else:
        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=requires_grad), input_color_data.size())

    # Rotate images clockwise
    if obj.use_cuda:
        rotate_color = F.grid_sample(Variable(input_color_data).cuda(), flow_grid_before, mode='nearest')
        rotate_depth = F.grid_sample(Variable(input_depth_data).cuda(), flow_grid_before, mode='nearest')
    else:
        rotate_color = F.grid_sample(Variable(input_color_data), flow_grid_before, mode='nearest')
        rotate_depth = F.grid_sample(Variable(input_depth_data), flow_grid_before, mode='nearest')

    # Compute intermediate features
    if efficientnet_pytorch is None:
        # densenet
        interm_push_color_feat = obj.push_color_trunk.features(rotate_color)
        interm_push_depth_feat = obj.push_depth_trunk.features(rotate_depth)
        interm_grasp_color_feat = obj.grasp_color_trunk.features(rotate_color)
        interm_grasp_depth_feat = obj.grasp_depth_trunk.features(rotate_depth)

        # TODO(hkwon214): added placenet to test block testing
        if obj.place:
            interm_place_color_feat = obj.place_color_trunk.features(rotate_color)
            interm_place_depth_feat = obj.place_depth_trunk.features(rotate_depth)
    else:
        # efficientnet
        interm_push_color_feat = obj.push_color_trunk.extract_features(rotate_color)
        interm_push_depth_feat = obj.push_depth_trunk.extract_features(rotate_depth)
        interm_grasp_color_feat = obj.grasp_color_trunk.extract_features(rotate_color)
        interm_grasp_depth_feat = obj.grasp_depth_trunk.extract_features(rotate_depth)

        # TODO(hkwon214): added placenet to test block testing
        if obj.place:
            interm_place_color_feat = obj.place_color_trunk.extract_features(rotate_color)
            interm_place_depth_feat = obj.place_depth_trunk.extract_features(rotate_depth)

    # Combine features, including the goal condition if appropriate
    if goal_condition is None:
        interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
        interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
        interm_place_feat = torch.cat((interm_place_color_feat, interm_place_depth_feat), dim=1)
        interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat, tiled_goal_condition), dim=1)
        interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat, tiled_goal_condition), dim=1)
        if obj.place:
            interm_place_feat = torch.cat((interm_place_color_feat, interm_place_depth_feat, tiled_goal_condition), dim=1)

    else:
        if obj.use_vector_block:
            push_goal_vec = tile_vector_as_image_channels_torch(obj.push_vector_block(goal_condition), interm_push_color_feat.shape)
            grasp_goal_vec = tile_vector_as_image_channels_torch(obj.grasp_vector_block(goal_condition), interm_push_color_feat.shape)
            interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat, push_goal_vec), dim=1)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat, grasp_goal_vec), dim=1)
            if obj.place:
                place_goal_vec = tile_vector_as_image_channels_torch(obj.place_vector_block(goal_condition), interm_push_color_feat.shape)
                interm_place_feat = torch.cat((interm_place_color_feat, interm_place_depth_feat, place_goal_vec), dim=1)

        else:
            if tiled_goal_condition is None:
                # This is part of a big for loop, but tiling only needs to be done once.
                # Sorry that this code is a bit confusing, but we need the shape of the output of interm_*_color_feat
                tiled_goal_condition = tile_vector_as_image_channels_torch(goal_condition, interm_push_color_feat.shape)
            interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat, tiled_goal_condition), dim=1)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat, tiled_goal_condition), dim=1)
            if obj.place:
                interm_place_feat = torch.cat((interm_place_color_feat, interm_place_depth_feat, tiled_goal_condition), dim=1)
    return interm_push_feat, interm_grasp_feat, interm_place_feat, tiled_goal_condition

class pixel_net(nn.Module):

    def __init__(self, use_cuda=True, goal_condition_len=0, place=False, network='efficientnet', use_vector_block=True): # , snapshot=None
        super(pixel_net, self).__init__()
        self.use_cuda = use_cuda
        self.place = place
        self.use_vector_block = use_vector_block

        if self.use_vector_block:
            channels_out = 2048
            self.push_vector_block = vector_block('push', goal_condition_len, channels_out=channels_out)
            self.grasp_vector_block = vector_block('grasp', goal_condition_len, channels_out=channels_out)
            if place:
                self.place_vector_block = vector_block('place', goal_condition_len, channels_out=channels_out)
            # TODO(ahundt) this variable overwrite is confusing, write the code better
            goal_condition_len = channels_out

        if network == 'densenet' or efficientnet_pytorch is None:
            # Initialize network trunks with DenseNet pre-trained on ImageNet
            self.push_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
            self.push_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
            self.grasp_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
            self.grasp_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)

            # TODO(hkwon214): added placenet to test block testing
            self.place_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
            self.place_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
            fc_channels = 2048
        else:
            # Initialize network trunks with DenseNet pre-trained on ImageNet
            self.push_color_trunk = EfficientNet.from_pretrained('efficientnet-b0')
            self.push_depth_trunk = EfficientNet.from_pretrained('efficientnet-b0')
            self.grasp_color_trunk = EfficientNet.from_pretrained('efficientnet-b0')
            self.grasp_depth_trunk = EfficientNet.from_pretrained('efficientnet-b0')

            # TODO(hkwon214): added placenet to test block testing
            self.place_color_trunk = EfficientNet.from_pretrained('efficientnet-b0')
            self.place_depth_trunk = EfficientNet.from_pretrained('efficientnet-b0')
            fc_channels = 1280 * 2

        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.pushnet = trunk_net('push', fc_channels, goal_condition_len, 1)
        self.graspnet = trunk_net('grasp', fc_channels, goal_condition_len, 1)
        # TODO(hkwon214): added placenet to test block testing
        if place:
            self.placenet = trunk_net('place', fc_channels, goal_condition_len, 1)

        # Initialize network weights
        for m in self.named_modules():
            #if 'push-' in m[0] or 'grasp-' in m[0]:
            #TODO(hkwon214): add condition 'if place:'
            if 'push-' in m[0] or 'grasp-' in m[0] or 'place-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()


        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []


    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1, goal_condition=None):

        if goal_condition is not None:
            # TODO(ahundt) is there a better place for this? Is doing this before is_volatile sloppy?
            if self.use_cuda:
                goal_condition = torch.tensor(goal_condition).float().cuda()
            else:
                goal_condition = torch.tensor(goal_condition).float()
            tiled_goal_condition = None

        if is_volatile:
            torch.set_grad_enabled(False)
            output_prob = []
            interm_feat = []

            # Apply rotations to images
            for rotate_idx in range(self.num_rotations):
                rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

                # Compute sample grid for rotation BEFORE neural network
                interm_push_feat, interm_grasp_feat, interm_place_feat, tiled_goal_condition = layers_forward(self, rotate_theta, input_color_data, input_depth_data, goal_condition)
                if self.place:
                    interm_feat.append([interm_push_feat, interm_grasp_feat, interm_place_feat])
                else:
                    interm_feat.append([interm_push_feat, interm_grasp_feat])

                # Compute sample grid for rotation AFTER branches
                affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_after.shape = (2,3,1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                if self.use_cuda:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_feat.data.size())
                else:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_feat.data.size())

                # Forward pass through branches, undo rotation on output predictions, upsample results
                # TODO(hkwon214): added placenet to test block testing
                if self.place:
                    output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                                    nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest')),
                                    nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).forward(F.grid_sample(self.placenet(interm_place_feat), flow_grid_after, mode='nearest'))])
                else:
                    output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                        nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest'))])

            torch.set_grad_enabled(True)
            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            interm_push_feat, interm_grasp_feat, interm_place_feat, tiled_goal_condition = layers_forward(self, rotate_theta, input_color_data, input_depth_data, goal_condition, tiled_goal_condition)
            if self.place:
                self.interm_feat.append([interm_push_feat, interm_grasp_feat, interm_place_feat])
            else:
                self.interm_feat.append([interm_push_feat, interm_grasp_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_feat.data.size())
            # print('goal_condition: ' + str(goal_condition))
            # Forward pass through branches, undo rotation on output predictions, upsample results
            # TODO(hkwon214): added placenet to test block testing
            if self.place:
                self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                                     nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest')),
                                     nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).forward(F.grid_sample(self.placenet(interm_place_feat), flow_grid_after, mode='nearest'))])
            else:
                self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                                     nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest'))])

            return self.output_prob, self.interm_feat