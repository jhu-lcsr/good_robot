import torch
from torch import nn as nn
import torch.nn.functional as F


class UResBlock(torch.nn.Module):
    def __init__(self, c_in=16):
        super(UResBlock, self).__init__()
        self.norm1 = nn.InstanceNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_in, 3, padding=1)
        self.act1 = nn.LeakyReLU()
        self.norm2 = nn.InstanceNorm2d(c_in)
        self.conv2 = nn.Conv2d(c_in, c_in, 3, padding=1)
        self.act2 = nn.LeakyReLU()

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


class UDeconvResBlock(torch.nn.Module):
    def __init__(self, c_in=16):
        super(UDeconvResBlock, self).__init__()
        self.norm1 = nn.InstanceNorm2d(c_in)
        self.deconv1 = nn.ConvTranspose2d(c_in, c_in, 3, padding=1)
        self.act1 = nn.LeakyReLU()
        self.norm2 = nn.InstanceNorm2d(c_in)
        self.deconv2 = nn.ConvTranspose2d(c_in, c_in, 3, padding=1)
        self.act2 = nn.LeakyReLU()

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.deconv1.weight)
        torch.nn.init.kaiming_uniform(self.deconv2.weight)

        self.deconv1.bias.data.fill_(0)
        self.deconv2.bias.data.fill_(0)

    def forward(self, images):
        # In case we pass in a Cx1x1 vector, we don't want to normalize that
        #if images.size(2) > 1:
        #    x = self.norm1(images)
        x = self.act1(self.deconv1(images, output_size=images.size()))
        x = self.act2(self.deconv2(self.norm2(x), output_size=x.size()))
        out = x + images
        return out


class Unet9Contextual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embedding_size, hc1=32, hc2=16, k=5, stride=2):
        super(Unet9Contextual, self).__init__()

        pad = int(k / 2)
        self.hc1 = hc1
        self.hc2 = hc2

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(in_channels, hc1, k, stride=stride, padding=pad)
        self.res1 = UResBlock(hc1)
        self.conv2 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.res2 = UResBlock(hc1)
        self.conv3 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.res3 = UResBlock(hc1)

        self.dropout = nn.Dropout(0.5)

        self.deres3 = UDeconvResBlock(hc1)
        self.deconv3 = nn.ConvTranspose2d(hc1, hc1, k, stride=stride, padding=pad)
        self.deres2 = UDeconvResBlock(hc1 * 2)
        self.deconv2 = nn.ConvTranspose2d(hc1 * 2, hc1, k, stride=stride, padding=pad)
        self.deres1 = UDeconvResBlock(hc1 * 2)
        self.deconv1 = nn.ConvTranspose2d(hc1 * 2, hc2, 3, stride=stride, padding=1)
        self.deconvout = nn.ConvTranspose2d(hc2, out_channels, k, stride=1, padding=pad)

        self.act = nn.LeakyReLU()

        self.lang1 = nn.Linear(embedding_size, hc1 * hc1)
        self.lang2 = nn.Linear(embedding_size, hc1 * hc1)
        self.lang3 = nn.Linear(embedding_size, hc1 * hc1)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv3.weight)
        self.conv3.bias.data.fill_(0)

        torch.nn.init.kaiming_uniform(self.deconv1.weight)
        self.deconv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv2.weight)
        self.deconv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv3.weight)
        self.deconv3.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconvout.weight)
        self.deconvout.bias.data.fill_(0)

        self.lang1.weight.data.normal_(0.001)
        self.lang2.weight.data.normal_(0.001)
        self.lang3.weight.data.normal_(0.001)

        self.lang1.bias.data.fill_(0)
        self.lang2.bias.data.fill_(0)
        self.lang3.bias.data.fill_(0)

    def forward(self, input, embedding):
        x1 = self.res1(self.act(self.conv1(input)))
        x2 = self.res2(self.act(self.conv2(x1)))
        x3 = self.res3(self.act(self.conv3(x2)))

        if embedding is not None:
            lf1 = F.normalize(self.lang1(embedding)).view([self.hc1, self.hc1, 1, 1])
            lf2 = F.normalize(self.lang2(embedding)).view([self.hc1, self.hc1, 1, 1])
            lf3 = F.normalize(self.lang3(embedding)).view([self.hc1, self.hc1, 1, 1])
            x1f = F.conv2d(x1, lf1)
            x2f = F.conv2d(x2, lf2)
            x3f = F.conv2d(x3, lf3)
            x1f = self.dropout(x1f)
            x2f = self.dropout(x2f)
            x3f = self.dropout(x3f)

        y2 = self.act(self.deconv3(self.deres3(x3f), output_size=x2.size()))
        y2in = torch.cat([x2f, y2], dim=1)

        y1 = self.act(self.deconv2(self.deres2(y2in), output_size=x1.size()))
        y1in = torch.cat([x1f, y1], dim=1)

        y0 = self.act(self.deconv1(self.deres1(y1in), output_size=input.size()))
        out = self.deconvout(y0, output_size=input.size())

        return out