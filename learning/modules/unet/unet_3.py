import torch
from torch import nn as nn
import torch.nn.functional as F

class Unet3(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embedding_size, hc1=32, hc2=16, k=5, stride=2):
        super(Unet3, self).__init__()

        pad = int(k / 2)
        self.hidden_channels = hc1

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(in_channels, hc1, k, stride=stride, padding=pad)
        self.conv2 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv3 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv4 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv5 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)

        self.dropout = nn.Dropout(0.5)

        self.deconv1 = nn.ConvTranspose2d(hc1, hc1, k, stride=stride, padding=pad)
        self.deconv2 = nn.ConvTranspose2d(2 * hc1, hc1, k, stride=stride, padding=pad)
        self.deconv3 = nn.ConvTranspose2d(2 * hc1, hc1, k, stride=stride, padding=pad)
        self.deconv4 = nn.ConvTranspose2d(2 * hc1, hc2, k, stride=stride, padding=pad)
        self.deconv5 = nn.ConvTranspose2d(hc1 + hc2, out_channels, k, stride=stride, padding=pad)

        self.norm2 = nn.InstanceNorm2d(hc1)
        self.norm3 = nn.InstanceNorm2d(hc1)
        self.norm4 = nn.InstanceNorm2d(hc1)
        self.norm5 = nn.InstanceNorm2d(hc1)
        #self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(hc1)
        self.dnorm3 = nn.InstanceNorm2d(hc1)
        self.dnorm4 = nn.InstanceNorm2d(hc1)
        self.dnorm5 = nn.InstanceNorm2d(hc2)

        self.act = nn.LeakyReLU()

        self.lang_filter_linear = nn.Linear(embedding_size, hc1 * hc1)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv5.weight)
        self.conv5.bias.data.fill_(0)

        torch.nn.init.kaiming_uniform(self.deconv1.weight)
        self.deconv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv2.weight)
        self.deconv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv3.weight)
        self.deconv3.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv4.weight)
        self.deconv4.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.deconv5.weight)
        self.deconv5.bias.data.fill_(0)

        self.lang_filter_linear.weight.normal_(0.001)
        self.lang_filter_linear.bias.fill_(0)

    def forward(self, input, embedding=None):
        x1 = self.norm2(self.act(self.conv1(input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if embedding is not None:
            lang_filter_weights = F.normalize(self.lang_filter_linear(embedding))
            lang_filter_weights = lang_filter_weights.view([self.hidden_channels, self.hidden_channels, 1, 1])
            x5 = F.conv2d(x5, lang_filter_weights)
            x5 = self.dropout(x5)

        x6 = self.act(self.deconv1(x5, output_size=x4.size()))
        x46 = torch.cat([x4, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1, x9], 1)
        out = self.deconv5(x19, output_size=input.size())

        """
        print("in:", input.size())
        print("x1:", x1.size())
        print("x2:", x2.size())
        print("x3:", x3.size())
        print("x4:", x4.size())
        print("x5:", x5.size())
        print("x6:", x6.size())
        """

        return out