import torch
import torch.nn as nn
from torch.nn import init

from learning.modules.identity import Identity
from learning.modules.rss.map_lang_semantic_filter import MapLangSemanticFilter

from learning.utils import layer_histogram_summaries


class ResBlockConditional(torch.nn.Module):
    def __init__(self, text_embed_size, channels=16, c_out=None):
        super(ResBlockConditional, self).__init__()
        if c_out is None:
            c_out = channels
        self.c_in = channels
        self.c_out = c_out
        if self.c_in != self.c_out:
            print("WARNING: ResBlockConditional is not residual")
        self.lf = MapLangSemanticFilter(text_embed_size, channels, c_out)

    def cuda(self, device=None):
        super(ResBlockConditional, self).cuda()
        self.lf.cuda(device)

    def init_weights(self):
        self.lf.init_weights()

    def forward(self, images, contexts):
        self.lf.precompute_conv_weights(contexts)
        x = self.lf(images)
        if self.c_in == self.c_out:
            out = x + images
        else:
            out = x
        return out


class ResBlock(torch.nn.Module):
    def __init__(self, c_in=16):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(c_in, c_in, 3, padding=1)
        self.conv2 = nn.Conv2d(c_in, c_in, 3, padding=1)

        self.act1 = nn.LeakyReLU()
        self.norm1 = nn.InstanceNorm2d(c_in)
        self.act2 = nn.LeakyReLU()
        self.norm2 = nn.InstanceNorm2d(c_in)

    def init_weights(self):
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)

        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)

    def forward(self, images):
        x = self.act1(self.conv1(self.norm1(images)))
        x = self.act2(self.conv2(self.norm2(x)))
        out = x + images
        return out

    def write_summaries(self, writer, prefix, idx):
        layer_histogram_summaries(writer, prefix + "/conv1", self.conv1, idx)
        layer_histogram_summaries(writer, prefix + "/conv2", self.conv2, idx)


class ResBlockStrided(torch.nn.Module):
    def __init__(self, c_in=16, stride=2, down_padding=0, groups=1, nonorm=False):
        super(ResBlockStrided, self).__init__()
        self.c_in = c_in
        self.conv1 = nn.Conv2d(c_in, c_in, 3, padding=down_padding, groups=groups)
        self.conv2 = nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=groups)

        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()

        if nonorm:
            self.norm1 = Identity()
            self.norm2 = Identity()
        else:
            self.norm1 = nn.InstanceNorm2d(c_in)
            self.norm2 = nn.InstanceNorm2d(c_in)

        self.avg_pool = nn.AvgPool2d(3, stride=stride, padding=down_padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)

        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)

    def forward(self, images):
        x = self.act1(self.conv1(self.norm1(images)))
        x_out = self.act2(self.conv2(self.norm2(x)))
        x_in = self.avg_pool(images)
        return x_in + x_out

    def write_summaries(self, writer, prefix, idx):
        layer_histogram_summaries(writer, prefix + "/conv1", self.conv1, idx)
        layer_histogram_summaries(writer, prefix + "/conv2", self.conv2, idx)


class ResBlockStridedConv(torch.nn.Module):
    def __init__(self, c_in=16, c_out=32, stride=2):
        super(ResBlockStridedConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm1 = nn.InstanceNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_in, 3, padding=1)
        self.act1 = nn.LeakyReLU()
        self.norm2 = nn.InstanceNorm2d(c_in)
        self.conv2 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)
        self.act2 = nn.LeakyReLU()

        self.convb = nn.Conv2d(c_in, c_out, 3, stride=stride)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        torch.nn.init.kaiming_uniform(self.conv2.weight)

        self.convb.weight.data.normal_(0, 0.0001)

        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)
        self.convb.bias.data.fill_(0)

    def forward(self, images):
        x = self.act1(self.conv1(self.norm1(images)))
        x_out = self.act2(self.conv2(self.norm2(x)))
        x_in = self.convb(images)
        return x_in + x_out


class ResBlockUp(torch.nn.Module):
    def __init__(self, c_in=16):
        super(ResBlockUp, self).__init__()
        self.norm1 = nn.InstanceNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_in, 1)
        self.act1 = nn.PReLU()
        self.norm2 = nn.InstanceNorm2d(c_in)
        self.conv2 = nn.Conv2d(c_in, c_in, 1)
        self.act2 = nn.PReLU()

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        torch.nn.init.kaiming_uniform(self.conv2.weight)

        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)

    def forward(self, images):
        x = self.act1(self.conv1(self.norm1(images)))
        x = self.act2(self.conv2(self.norm2(x)))
        out = x + images
        return out


class UpsampleBlock(torch.nn.Module):
    def __init__(self, c_in, c_out, upscale_factor):
        super(UpsampleBlock, self).__init__()

        self.prelu = nn.PReLU()
        self.conv1 = nn.Conv2d(c_in, c_out * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(c_out)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        #x = self.prelu(x)
        return x

    def init_weights(self):
        torch.nn.init.orthogonal(self.conv1.weight, torch.nn.init.calculate_gain('relu'))


class DenseBlock(torch.nn.Module):
    def __init__(self, c_in=32, c_out=32):
        super(DenseBlock, self).__init__()
        self.norm1 = nn.InstanceNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.act1 = nn.LeakyReLU()
        self.norm2 = nn.InstanceNorm2d(c_in + c_out)
        self.conv2 = nn.Conv2d(c_in + c_out, c_in, 3, padding=1)
        self.act2 = nn.LeakyReLU()
        self.norm3 = nn.InstanceNorm2d(c_in + 2 * c_out)
        self.conv3 = nn.Conv2d(c_in + 2 * c_out, c_out, 3, padding=1)
        self.act3 = nn.LeakyReLU()

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        torch.nn.init.kaiming_uniform(self.conv2.weight)

        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)

    def forward(self, images):
        x1 = self.act1(self.conv1(self.norm1(images)))
        #print(x1.size(), images.size())
        x1_cat = torch.cat([images, x1], dim=1)
        x2 = self.act2(self.conv2(self.norm2(x1_cat)))
        x2_cat = torch.cat([images, x1, x2], dim=1)
        x3 = self.act3(self.conv3(self.norm3(x2_cat)))
        return x3


class DenseMlpBlock2(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(DenseMlpBlock2, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(in_size + hidden_size, out_size)
        self.act1 = nn.LeakyReLU()

    def init_weights(self):
        init.orthogonal(self.linear1.weight, init.calculate_gain("leaky_relu"))
        init.orthogonal(self.linear2.weight, init.calculate_gain("leaky_relu"))
        #self.linear1.weight.data.normal_(0, 0.001)
        #self.linear2.weight.data.normal_(0, 0.001)
        self.linear1.bias.data.fill_(0)
        self.linear2.bias.data.fill_(0)

    def forward(self, input):
        x1 = self.act1(self.linear1(input))
        x1_cat = torch.cat([input, x1], dim=1)
        x2 = self.linear2(x1_cat)
        return x2

    def write_summaries(self, writer, prefix, idx):
        layer_histogram_summaries(writer, prefix + "/linear1", self.linear1, idx)
        layer_histogram_summaries(writer, prefix + "/linear2", self.linear2, idx)


class DenseMlpBlock3(torch.nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super(DenseMlpBlock3, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(in_size + hidden_size, hidden_size)
        self.linear3 = nn.Linear(in_size + 2*hidden_size, out_size)
        self.act1 = nn.LeakyReLU()

    def init_weights(self):
        #init.orthogonal(self.linear1.weight, init.calculate_gain("leaky_relu"))
        #init.orthogonal(self.linear2.weight, init.calculate_gain("leaky_relu"))
        #init.orthogonal(self.linear3.weight, init.calculate_gain("leaky_relu"))
        self.linear1.weight.data.normal_(0, 0.001)
        self.linear2.weight.data.normal_(0, 0.001)
        self.linear3.weight.data.normal_(0, 0.001)
        self.linear1.bias.data.fill_(0)
        self.linear2.bias.data.fill_(0)
        self.linear3.bias.data.fill_(0)

    def forward(self, input):
        x1 = self.act1(self.linear1(input))
        x1_cat = torch.cat([input, x1], dim=1)
        x2 = self.act1(self.linear2(x1_cat))
        x2_cat = torch.cat([input, x1, x2], dim=1)
        x3 = self.linear3(x2_cat)
        return x3

    def write_summaries(self, writer, prefix, idx):
        layer_histogram_summaries(writer, prefix + "/linear1", self.linear1, idx)
        layer_histogram_summaries(writer, prefix + "/linear2", self.linear2, idx)
        layer_histogram_summaries(writer, prefix + "/linear3", self.linear3, idx)
