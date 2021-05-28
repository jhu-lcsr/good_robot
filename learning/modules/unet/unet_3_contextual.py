import torch
from torch import nn as nn
import torch.nn.functional as F

class Unet3Contextual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embedding_size, hc1=32, hc2=16, k=5, stride=2):
        super(Unet3Contextual, self).__init__()

        pad = int(k / 2)
        self.hc1 = hc1
        self.hc2 = hc2

        # inchannels, outchannels, kernel size

        self.conv1 = nn.Conv2d(in_channels, hc1, k, stride=stride, padding=pad)
        self.conv2 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv3 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)

        self.dropout = nn.Dropout(0.5)

        self.deconv1 = nn.ConvTranspose2d(hc1, hc1, k, stride=stride, padding=pad)
        self.deconv2 = nn.ConvTranspose2d(2 * hc1, hc2, k, stride=stride, padding=pad)
        self.deconv3 = nn.ConvTranspose2d(hc1 + hc2, out_channels, k, stride=stride, padding=pad)

        self.norm2 = nn.InstanceNorm2d(hc1)
        self.norm3 = nn.InstanceNorm2d(hc1)
        #self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(hc1)
        self.dnorm3 = nn.InstanceNorm2d(hc2)

        self.act = nn.LeakyReLU()

        self.lang11 = nn.Linear(embedding_size, hc1 * hc1)
        self.lang22 = nn.Linear(embedding_size, hc1 * hc1)
        self.lang33 = nn.Linear(embedding_size, hc1 * hc1)

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

        self.lang11.weight.data.normal_(0.001)
        self.lang22.weight.data.normal_(0.001)
        self.lang33.weight.data.normal_(0.001)
        self.lang11.bias.data.fill_(0)
        self.lang22.bias.data.fill_(0)
        self.lang33.bias.data.fill_(0)

    def forward(self, input, embedding):
        x1 = self.norm2(self.act(self.conv1(input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.act(self.conv3(x2))

        if embedding is not None:
            lf1 = F.normalize(self.lang11(embedding)).view([self.hc1, self.hc1, 1, 1])
            lf2 = F.normalize(self.lang22(embedding)).view([self.hc1, self.hc1, 1, 1])
            lf3 = F.normalize(self.lang33(embedding)).view([self.hc1, self.hc1, 1, 1])
            x1f = F.conv2d(x1, lf1)
            x2f = F.conv2d(x2, lf2)
            x3f = F.conv2d(x3, lf3)
            x3f = self.dropout(x3f)
            x2f = self.dropout(x2f)
            x1f = self.dropout(x1f)

        y2 = self.act(self.deconv1(x3f, output_size=x2.size()))
        y22 = torch.cat([y2, x2f], 1)
        y1 = self.dnorm3(self.act(self.deconv2(y22, output_size=x1.size())))
        y11 = torch.cat([y1, x1f], 1)
        out = self.deconv3(y11, output_size=input.size())

        return out