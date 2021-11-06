import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable

from learning.inputs.common import empty_float_tensor


class MapLangSemanticFilter(nn.Module):
    def __init__(self, text_embed_size, in_map_channels, out_map_channels):
        super(MapLangSemanticFilter, self).__init__()

        self.num_conv_weights = in_map_channels * out_map_channels
        self.in_map_channels = in_map_channels
        self.out_map_channels = out_map_channels

        self.lang_gate_linear = nn.Linear(text_embed_size, self.num_conv_weights)

        self.conv_weights = None

        self.norm = nn.InstanceNorm2d(out_map_channels)

    def precompute_conv_weights(self, text_embeddings):
        batch_size = text_embeddings.size(0)
        self.conv_weights = torch.sigmoid(self.lang_gate_linear(text_embeddings))
        self.conv_weights = F.normalize(self.conv_weights, dim=1)
        self.conv_weights = self.conv_weights.view(batch_size, self.out_map_channels, self.in_map_channels, 1, 1)

    def init_weights(self):
        self.lang_gate_linear.weight.data.normal_(0, 0.001)
        self.lang_gate_linear.bias.data.fill_(0)

    def forward(self, input):
        input_batch_size = input.size(0)
        weight_batch_size = self.conv_weights.size(0)
        batch_factor = int(input_batch_size / weight_batch_size)

        out_size = list(input.size())
        out_size[1] = self.out_map_channels
        out = torch.zeros(out_size).to(input.device)

        for group in range(weight_batch_size):
            group_inputs = input[group*batch_factor:(group+1)*batch_factor]
            conv_w_group = self.conv_weights[group]
            x = F.conv2d(group_inputs, conv_w_group, dilation=1, padding=0)
            x = F.leaky_relu(x)
            out[group*batch_factor:(group+1)*batch_factor] = x

        # TODO: Test and consider
        out = self.norm(out)

        return out