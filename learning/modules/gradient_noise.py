import torch
import numpy as np
from torch import nn as nn


class GradientNoise(nn.Module):
    def __init__(self):
        super(GradientNoise, self).__init__()

    def forward(self, x):
        x.register_hook(self.backward_hook)
        return x

    def backward_hook(self, grad_input):
        # grad_input has shape Bx2x64x64
        batch_size = grad_input.shape[0]

        # Should we scale per-dimension?
        mean_grad = grad_input.view(batch_size, -1).mean(1)
        sigma = 0.5 * mean_grad
        standard_normal_noise = torch.randn(grad_input.shape).to(grad_input.device)
        #print("GRADBACK: ", standard_normal_noise.shape, grad_input.shape, sigma.shape)

        noise = standard_normal_noise * sigma[:, np.newaxis, np.newaxis, np.newaxis]

        noisy_grad_input = grad_input + noise
        return noisy_grad_input