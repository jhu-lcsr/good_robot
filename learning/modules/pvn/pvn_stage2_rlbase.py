import torch
import numpy as np
import torch.nn as nn

from visualization import Presenter


class PVN_Stage2_RLBase(nn.Module):
    def __init__(self, map_channels=1, map_struct_channels=1, map_size=32, crop_size=16, h1=8, structure_h1=8, h2=128, obs_dim=16, name="base"):
        super(PVN_Stage2_RLBase, self).__init__()

        self.map_channels = map_channels
        self.map_structure_channels = map_struct_channels
        self.map_size = map_size
        self.crop_size = crop_size
        self.name = name

        gap = int((map_size - crop_size) / 2)
        self.crop_l = gap
        self.crop_r = map_size - gap

        self.conv1 = nn.Conv2d(map_channels, h1, kernel_size=3, stride=2, padding=1, bias=True)

        self.structconv1 = nn.Conv2d(self.map_structure_channels, structure_h1, kernel_size=3, stride=2, padding=1, bias=True)

        linear_in = int(((self.crop_size / 2) ** 2) * h1) + (8*8*structure_h1) + 2*obs_dim # map channels + coverage channels + observability encoding channels
        print(f"Stage 2 linear input size: {linear_in}")
        self.linear1 = nn.Linear(linear_in, h2)
        self.linear2 = nn.Linear(h2 + linear_in, h2)

        self.goal_in_vec = nn.Parameter(torch.Tensor(obs_dim))
        self.goal_out_vec = nn.Parameter(torch.Tensor(obs_dim))
        self.visit_in_vec = nn.Parameter(torch.Tensor(obs_dim))
        self.visit_out_vec = nn.Parameter(torch.Tensor(obs_dim))

        self.avgpool = nn.AvgPool2d(4)

        self.act = nn.LeakyReLU()
        self.norm1 = nn.InstanceNorm2d(h1)
        self.covnorm1 = nn.InstanceNorm2d(structure_h1)

    def init_weights(self):
        self.goal_in_vec.data.normal_(0, 1.0)
        self.goal_out_vec.data.copy_(-self.goal_in_vec.data)
        self.visit_in_vec.data.normal_(0, 1.0)
        self.visit_out_vec.data.copy_(-self.visit_in_vec.data)

    def backward_hook(self, name, grad):
        if False:
            print(f"Grad stats {self.name}/{name}: {grad.max()} {grad.min()} {grad.mean()}")

    def forward(self, maps_r, map_structure_r):
        maps_r_cropped = maps_r.inner_distribution[:, :, self.crop_l:self.crop_r, self.crop_l:self.crop_r]
        batch_size = maps_r.inner_distribution.shape[0]

        # Create a context vector that encodes goal observability
        # Don't backprop into the embedding vectors - don't risk losing the only input we have
        gin = self.goal_in_vec.detach()[np.newaxis, :].repeat([batch_size, 1])
        gout = self.goal_out_vec.detach()[np.newaxis, :].repeat([batch_size, 1])
        vin = self.visit_in_vec.detach()[np.newaxis, :].repeat([batch_size, 1])
        vout = self.visit_out_vec.detach()[np.newaxis, :].repeat([batch_size, 1])

        p_visit_out = maps_r.outer_prob_mass[:, 0:1].detach()
        p_goal_out = maps_r.outer_prob_mass[:, 1:2].detach()

        g_context_vec = gout * p_goal_out + gin * (1 - p_goal_out)
        v_context_vec = vout * p_visit_out + vin * (1 - p_visit_out)
        obs_context_vec = torch.cat([g_context_vec, v_context_vec], dim=1)

        # 64x64 -> 16x16
        uncov_r_pooled = self.avgpool(map_structure_r)

        if False:
            conv_in_np = conv_in[0].data.cpu().numpy().transpose(1, 2, 0)
            # expand to 0-1 range
            conv_in_np[:, :, 0] /= (np.max(conv_in_np[:, :, 0]) + 1e-10)
            conv_in_np[:, :, 1] /= (np.max(conv_in_np[:, :, 1]) + 1e-10)
            conv_in_np[:, :, 2] /= (np.max(conv_in_np[:, :, 2]) + 1e-10)
            Presenter().show_image(conv_in_np, "rl_conv_in", scale=2)
            #Presenter().show_image(uncov_r_pooled[0], "uncov_pooled", scale=4)

        # From 16x16 down to 8x8
        x = self.act(self.conv1(maps_r_cropped))
        x = self.norm1(x)

        # From 16x16 down to 8x8
        c = self.act(self.structconv1(uncov_r_pooled))
        c = self.covnorm1(c)

        comb_map = torch.cat([x, c], dim=1)
        batch_size = x.shape[0]
        lin_in = comb_map.view(batch_size, -1)
        lin_in = torch.cat([lin_in, obs_context_vec], dim=1)

        x = self.act(self.linear1(lin_in))
        x = torch.cat([lin_in, x], dim=1)
        x = self.act(self.linear2(x))

        return x
