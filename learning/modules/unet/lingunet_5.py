import torch
from torch.autograd import Variable
from torch import nn as nn
import torch.nn.functional as F

from learning.inputs.partial_2d_distribution import Partial2DDistribution

from utils.dict_tools import objectview


class DoubleConv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(cin, cin, k, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(cin, cout, k, stride=1, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img):
        x = self.conv1(img)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        return x


class DoubleDeconv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super(DoubleDeconv, self).__init__()
        self.conv1 = nn.ConvTranspose2d(cin, cout, k, stride=1, padding=padding)
        self.conv2 = nn.ConvTranspose2d(cout, cout, k, stride=stride, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img, output_size):
        # TODO: 2 is stride
        osize1 = [int(i/2) for i in output_size]
        x = self.conv1(img, output_size=osize1)
        x = F.leaky_relu(x)
        x = self.conv2(x, output_size=output_size)
        return x

class Lingunet5(torch.nn.Module):
    def __init__(self, params):
                 #in_channels, out_channels, embedding_size, hc1=32, hb1=16, hc2=256, stride=2, split_embedding=False):
        super(Lingunet5, self).__init__()

        self.p = objectview(params)

        if self.p.split_embedding:
            self.emb_block_size = int(self.p.embedding_size / 5)
        else:
            self.emb_block_size = self.p.embedding_size

        # inchannels, outchannels, kernel size
        self.conv1 = DoubleConv(self.p.in_channels, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv2 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv3 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv4 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv5 = DoubleConv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)

        self.deconv1 = DoubleDeconv(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv2 = DoubleDeconv(self.p.hc1 + self.p.hb1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv3 = DoubleDeconv(self.p.hc1 + self.p.hb1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv4 = DoubleDeconv(self.p.hc1 + self.p.hb1, self.p.hc2, 3, stride=self.p.stride, padding=1)
        self.deconv5 = nn.ConvTranspose2d(self.p.hb1 + self.p.hc2, self.p.out_channels, 3, stride=self.p.stride, padding=1)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(self.p.hc1)
        self.norm3 = nn.InstanceNorm2d(self.p.hc1)
        self.norm4 = nn.InstanceNorm2d(self.p.hc1)
        self.norm5 = nn.InstanceNorm2d(self.p.hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm3 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm4 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm5 = nn.InstanceNorm2d(self.p.hc2)

        self.fnorm1 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm2 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm3 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm4 = nn.InstanceNorm2d(self.p.hb1)

        self.lang19 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang28 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang37 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang46 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang55 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hc1)

    def init_weights(self):
        self.conv1.init_weights()
        self.conv2.init_weights()
        self.conv3.init_weights()
        self.conv4.init_weights()
        self.conv5.init_weights()
        self.deconv1.init_weights()
        self.deconv2.init_weights()
        self.deconv3.init_weights()
        self.deconv4.init_weights()
        #self.deconv5.init_weights()

    def forward(self, input, embedding, tensor_store=None):
        x1 = self.norm2(self.act(self.conv1(input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        if tensor_store is not None:
            tensor_store.keep_inputs("lingunet_f1", x1)
            tensor_store.keep_inputs("lingunet_f2", x2)
            tensor_store.keep_inputs("lingunet_f3", x3)
            tensor_store.keep_inputs("lingunet_f4", x4)
            tensor_store.keep_inputs("lingunet_f5", x5)

        if embedding is not None:
            embedding = F.normalize(embedding, p=2, dim=1)

            if self.p.split_embedding:
                block_size = self.emb_block_size
                emb1 = embedding[:, 0*block_size:1*block_size]
                emb2 = embedding[:, 1*block_size:2*block_size]
                emb3 = embedding[:, 2*block_size:3*block_size]
                emb4 = embedding[:, 3*block_size:4*block_size]
                emb5 = embedding[:, 4*block_size:5*block_size]
            else:
                emb1 = emb2 = emb3 = emb4 = emb5 = embedding

            # These conv filters are different for each element in the batch, but the functional convolution
            # operator assumes the same filters across the batch.
            # TODO: Verify if slicing like this is a terrible idea for performance
            x1f = Variable(torch.zeros_like(x1[:,0:self.p.hb1,:,:].data))
            x2f = Variable(torch.zeros_like(x2[:,0:self.p.hb1,:,:].data))
            x3f = Variable(torch.zeros_like(x3[:,0:self.p.hb1,:,:].data))
            x4f = Variable(torch.zeros_like(x4[:,0:self.p.hb1,:,:].data))
            x5f = Variable(torch.zeros_like(x5.data))

            batch_size = input.shape[0]
            for i in range(batch_size):
                emb_idx = i if embedding.shape[0] == batch_size else 0

                lf1 = F.normalize(self.lang19(emb1[emb_idx:emb_idx + 1])).view([self.p.hb1, self.p.hc1, 1, 1])
                lf2 = F.normalize(self.lang28(emb2[emb_idx:emb_idx + 1])).view([self.p.hb1, self.p.hc1, 1, 1])
                lf3 = F.normalize(self.lang37(emb3[emb_idx:emb_idx + 1])).view([self.p.hb1, self.p.hc1, 1, 1])
                lf4 = F.normalize(self.lang46(emb4[emb_idx:emb_idx + 1])).view([self.p.hb1, self.p.hc1, 1, 1])
                lf5 = F.normalize(self.lang55(emb5[emb_idx:emb_idx + 1])).view([self.p.hc1, self.p.hc1, 1, 1])

                # Dropout on the convolutional filters computed from the language embedding. This might be a bad idea?
                #lf1 = self.dropout(lf1)
                #lf2 = self.dropout(lf2)
                #lf3 = self.dropout(lf3)
                #lf4 = self.dropout(lf4)
                #lf5 = self.dropout(lf5)

                x1f[i:i+1] = F.conv2d(x1[i:i+1], lf1)
                x2f[i:i+1] = F.conv2d(x2[i:i+1], lf2)
                x3f[i:i+1] = F.conv2d(x3[i:i+1], lf3)
                x4f[i:i+1] = F.conv2d(x4[i:i+1], lf4)
                x5f[i:i+1] = F.conv2d(x5[i:i+1], lf5)

            x1 = self.fnorm1(x1f)
            x2 = self.fnorm2(x2f)
            x3 = self.fnorm3(x3f)
            x4 = self.fnorm4(x4f)
            x5 = x5f

            if tensor_store is not None:
                tensor_store.keep_inputs("lingunet_g1", x1)
                tensor_store.keep_inputs("lingunet_g2", x2)
                tensor_store.keep_inputs("lingunet_g3", x3)
                tensor_store.keep_inputs("lingunet_g4", x4)
                tensor_store.keep_inputs("lingunet_g5", x5)

            # Dropout on the feature maps computed after filtering the input feature maps.
            #x1 = self.dropout2(x1)
            #x2 = self.dropout2(x2)
            #x3 = self.dropout2(x3)
            #x4 = self.dropout2(x4)
            #x5 = self.dropout2(x5)

        x6 = self.act(self.deconv1(x5, output_size=x4.size()))
        x46 = torch.cat([x4, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1, x9], 1)
        inner_scores = self.deconv5(x19, output_size=input.size())

        outer_scores = torch.zeros_like(inner_scores[:,:,0,0])
        both_dist_scores = Partial2DDistribution(inner_scores, outer_scores)

        return both_dist_scores