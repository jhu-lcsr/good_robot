from typing import Tuple, List, Dict

import torch

class ImageEncoder(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 n_layers: int,
                 activation: str = "relu", 
                 factor: int = 4, 
                 dropout: float = 0.2):
        super(ImageEncoder, self).__init__()

        self.input_dim = input_dim
        self.factor = factor
        # will be set later 
        self.device = torch.device("cpu") 

        output_dim = 2

        self.n_layers = n_layers
        self.activation = torch.nn.ReLU() if activation == "relu" else torch.nn.Tanh()
        self.dropout = dropout

        layers = []
        factor = 2
        kernel_size = 3
        maxpool_kernel_size = 3
        stride = 1
        output_wh = 64

        for i in range(self.n_layers):
            layers.append(torch.nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride, padding = 0))
            output_wh = (output_wh - (kernel_size-1)) 
            layers.append(self.activation)
            layers.append(torch.nn.MaxPool2d(maxpool_kernel_size))
            output_wh = int((output_wh - (maxpool_kernel_size-1) - 1)/ maxpool_kernel_size) + 1

            layers.append(torch.nn.Dropout2d(p = self.dropout))
            #input_dim = output_dim
            #output_dim = max(16, int(input_dim / factor)) 
       
        # infer output size 
        #self.output_dim = output_dim * output_wh**3
        self.output_dim = output_wh * output_wh * output_dim 
        self.layers = torch.nn.ModuleList(layers) 

    def forward(self, inputs):
        bsz,  width, height, n_labels = inputs.shape
        #outputs = inputs.reshape(bsz, n_labels depth, height, width)
        outputs = inputs.to(self.device)
        for layer in self.layers:
            outputs = layer(outputs) 

        # flatten; will error out if shape inference was wrong 
        #outputs = outputs.reshape((bsz, 1, self.output_dim))
        # flatten 
        outputs = outputs.reshape((bsz, 1, -1))
        return outputs 


def infer_kernel_size(input_dim,
                      output_dim,
                      stride = 1, 
                      input_padding = 0,
                      output_padding = 0,
                      dilation = 1):
    num = output_dim - (input_dim - 1) * stride + 2 * input_padding - output_padding - 1
    frac = num/dilation + 1
    return int(frac)

def get_deconv_out_dim(input_dim, 
                      kernel,
                      stride = 1, 
                      input_padding = 0,
                      output_padding = 0,
                      dilation = 1):
    return (input_dim - 1) * stride - 2 * input_padding + dilation * (kernel - 1) + output_padding + 1

class DeconvolutionalNetwork(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_blocks: int,
                 num_layers: int = 3,
                 factor: int = 2,
                 dropout: float = 0.2): 
        super(DeconvolutionalNetwork, self).__init__() 
        self.input_channels = input_channels
        self.factor = factor
        self.num_layers = num_layers
        # will be set later 
        self.device = torch.device("cpu") 

        self.activation = torch.nn.ReLU() 
        self.dropout = torch.nn.Dropout3d(p=dropout) 
        layers = []

        #kernel_size = (int(64/(self.num_layers-1)), int(64/(self.num_layers - 1)), 2) #max(1, int(4/self.num_layers)))
        #kernel_size = 4
        xy_input_dim = 1
        z_input_dim = 1
        xy_output_dim = max(1, int(64/self.num_layers))
        z_output_dim = max(1, int(4/self.num_layers))
        output_channels = max(1, int(input_channels/2)) 
        
        for i in range(num_layers):
            xy_kernel = infer_kernel_size(xy_input_dim,xy_output_dim)  
            z_kernel = infer_kernel_size(z_input_dim, z_output_dim)
            kernel_size = [xy_kernel, xy_kernel, z_kernel]

            layers.append(torch.nn.ConvTranspose3d(input_channels, output_channels, kernel_size, padding=0)) 
            layers.append(self.activation) 
            layers.append(self.dropout) 
            xy_input_dim = xy_output_dim
            z_input_dim = z_output_dim
            xy_output_dim += xy_output_dim
            z_output_dim += z_output_dim
            input_channels = output_channels
            output_channels = max(1, int(output_channels/2)) 

        self.output_dim = xy_output_dim
        # per pixel per class
        conv_last = torch.nn.ConvTranspose3d(output_channels*2, num_blocks+1, 1, padding=0)
        layers.append(conv_last) 
        self.layers = torch.nn.ModuleList(layers) 

    def forward(self, encoded):
        encoded = encoded.to(self.device) 
        # encoded: [batch, seq_len, input_dim]
        bsz, input_dim = encoded.data.shape
        # reshape [bsz, 4, 4, input_dim/8]
        encoded = encoded.reshape((bsz, -1, 1, 1, 1))

        for layer in self.layers:
            encoded = layer(encoded) 
        # output: [batch, width, height] 
        output = encoded.reshape((bsz, 21, 64, 64, 4))
        return output 
