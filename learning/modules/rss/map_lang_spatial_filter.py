import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable

from learning.inputs.common import empty_float_tensor


class MapLangSpatialFilter(torch.nn.Module):
    def __init__(self, text_embed_size, in_map_channels, out_map_channels):
        super(MapLangSpatialFilter, self).__init__()

        self.num_conv1_weights = in_map_channels * (3 * 3) * in_map_channels
        self.num_conv2_weights = in_map_channels * (3 * 3) * out_map_channels

        self.in_map_channels = in_map_channels
        self.out_map_channels = out_map_channels

        self.lang_gate_linear = nn.Linear(text_embed_size, self.num_conv1_weights + self.num_conv2_weights)

        self.conv1_weights = None
        self.conv2_weights = None

        self.norm = nn.InstanceNorm2d(in_map_channels)
        self.norm_out = nn.InstanceNorm2d(out_map_channels)

    def precompute_conv_weights(self, text_embedding):
        batch_size = text_embedding.size(0)
        all_weights = torch.sigmoid(self.lang_gate_linear(text_embedding))

        self.conv1_weights = all_weights[:, 0:self.num_conv1_weights]
        self.conv2_weights = all_weights[:, self.num_conv1_weights:self.num_conv1_weights+self.num_conv2_weights]

        self.conv1_weights = F.normalize(self.conv1_weights, dim=1)
        self.conv1_weights = self.conv1_weights.view(batch_size, self.in_map_channels, self.in_map_channels, 3, 3)

        self.conv2_weights = F.normalize(self.conv2_weights, dim=1)
        self.conv2_weights = self.conv2_weights.view(batch_size, self.out_map_channels, self.in_map_channels, 3, 3)

    def init_weights(self):
        self.lang_gate_linear.weight.data.normal_(0, 0.001)
        self.lang_gate_linear.bias.data.fill_(0)

    def forward(self, input):
        input_batch_size = input.size(0)
        weight_batch_size = self.conv1_weights.size(0)
        batch_factor = int(input_batch_size / weight_batch_size)

        out_size = list(input.size())
        out_size[1] = self.out_map_channels
        out = torch.zeros(out_size).to(input.device)

        for group in range(weight_batch_size):
            group_inputs = input[group*batch_factor:(group+1)*batch_factor]
            conv1_w_group = self.conv1_weights[group]
            conv2_w_group = self.conv2_weights[group]
            x = F.conv2d(group_inputs, conv1_w_group, dilation=1, padding=1)
            x = F.leaky_relu(x)
            x = self.norm(x)
            x = F.conv2d(x, conv2_w_group, dilation=3, padding=3)
            out[group*batch_factor:(group+1)*batch_factor] = x

        out = F.leaky_relu(out)
        out = self.norm_out(out)

        return out