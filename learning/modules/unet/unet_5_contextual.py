import torch
from torch.autograd import Variable
from torch import nn as nn
import torch.nn.functional as F


class Unet5Contextual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, embedding_size, hc1=32, hc2=256, k=5, stride=2, split_embedding=False):
        super(Unet5Contextual, self).__init__()

        pad = int(k / 2)
        self.hc1 = hc1
        self.hc2 = hc2

        self.split_embedding = split_embedding

        self.embedding_size = embedding_size
        if split_embedding:
            self.emb_block_size = int(embedding_size / 5)
        else:
            self.emb_block_size = embedding_size

        # inchannels, outchannels, kernel size
        self.conv1 = nn.Conv2d(in_channels, hc1, k, stride=stride, padding=pad)
        self.conv2 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv3 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv4 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)
        self.conv5 = nn.Conv2d(hc1, hc1, k, stride=stride, padding=pad)

        self.deconv1 = nn.ConvTranspose2d(hc1, hc1, k, stride=stride, padding=pad)
        self.deconv2 = nn.ConvTranspose2d(2 * hc1, hc1, k, stride=stride, padding=pad)
        self.deconv3 = nn.ConvTranspose2d(2 * hc1, hc1, k, stride=stride, padding=pad)
        self.deconv4 = nn.ConvTranspose2d(2 * hc1, hc2, k, stride=stride, padding=pad)
        self.deconv5 = nn.ConvTranspose2d(hc1 + hc2, out_channels, k, stride=stride, padding=pad)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(hc1)
        self.norm3 = nn.InstanceNorm2d(hc1)
        self.norm4 = nn.InstanceNorm2d(hc1)
        self.norm5 = nn.InstanceNorm2d(hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(hc1)
        self.dnorm3 = nn.InstanceNorm2d(hc1)
        self.dnorm4 = nn.InstanceNorm2d(hc1)
        self.dnorm5 = nn.InstanceNorm2d(hc2)

        self.lang19 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang28 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang37 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang46 = nn.Linear(self.emb_block_size, hc1 * hc1)
        self.lang55 = nn.Linear(self.emb_block_size, hc1 * hc1)

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

    def forward(self, input, embedding):
        x1 = self.norm2(self.act(self.conv1(input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if embedding is not None:
            if self.split_embedding:
                block_size = self.emb_block_size
                emb1 = embedding[:, 0*block_size:1*block_size]
                emb2 = embedding[:, 1*block_size:2*block_size]
                emb3 = embedding[:, 2*block_size:3*block_size]
                emb4 = embedding[:, 3*block_size:4*block_size]
                emb5 = embedding[:, 4*block_size:5*block_size]
            else:
                emb1 = emb2 = emb3 = emb4 = emb5 = embedding

            lf1 = F.normalize(self.lang19(emb1)).view([self.hc1, self.hc1, 1, 1])
            lf2 = F.normalize(self.lang28(emb2)).view([self.hc1, self.hc1, 1, 1])
            lf3 = F.normalize(self.lang37(emb3)).view([self.hc1, self.hc1, 1, 1])
            lf4 = F.normalize(self.lang46(emb4)).view([self.hc1, self.hc1, 1, 1])
            lf5 = F.normalize(self.lang55(emb5)).view([self.hc1, self.hc1, 1, 1])

            # Dropout on the convolutional filters computed from the language embedding. This might be a bad idea?
            # testing without dropout 4/16
            #lf1 = self.dropout(lf1)
            #lf2 = self.dropout(lf2)
            #lf3 = self.dropout(lf3)
            #lf4 = self.dropout(lf4)
            #lf5 = self.dropout(lf5)

            x1f = F.conv2d(x1, lf1)
            x2f = F.conv2d(x2, lf2)
            x3f = F.conv2d(x3, lf3)
            x4f = F.conv2d(x4, lf4)
            x5f = F.conv2d(x5, lf5)

            # Dropout on the feature maps computed after filtering the input feature maps.
            x1f = self.dropout2(x1f)
            x2f = self.dropout2(x2f)
            x3f = self.dropout2(x3f)
            x4f = self.dropout2(x4f)
            x5f = self.dropout2(x5f)

        x6 = self.act(self.deconv1(x5f, output_size=x4.size()))
        x46 = torch.cat([x4f, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3f, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2f, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1f, x9], 1)
        out = self.deconv5(x19, output_size=input.size())

        return out