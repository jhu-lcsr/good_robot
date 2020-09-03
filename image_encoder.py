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
        #output_dim = int(input_dim / factor) 
        output_dim = 1

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
            print(input_dim, output_dim) 
            layers.append(torch.nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride, padding = 0))
            output_wh = (output_wh - (kernel_size-1)) 
            layers.append(self.activation)
            layers.append(torch.nn.MaxPool2d(maxpool_kernel_size))
            output_wh = int((output_wh - (maxpool_kernel_size-1) - 1)/ maxpool_kernel_size) + 1

            layers.append(torch.nn.Dropout2d(p = self.dropout))
            #input_dim = output_dim
            output_dim = max(16, int(input_dim / factor)) 
       
        # infer output size 
        self.output_dim = output_dim * output_wh**3
        self.layers = torch.nn.ModuleList(layers) 

    def forward(self, inputs):
        print(f"input image {inputs.shape}") 
        bsz,  width, height, n_labels = inputs.shape
        #outputs = inputs.reshape(bsz, n_labels depth, height, width)
        outputs = inputs 
        for layer in self.layers:
            outputs = layer(outputs) 

        # flatten; will error out if shape inference was wrong 
        outputs = outputs.reshape((bsz, 1, self.output_dim))
        return outputs 

class DeconvolutionalNetwork(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_blocks: int,
                 num_layers: int = 3,
                 factor: int = 2,
                 dropout: float = 0.2): 
        super(DeconvolutionalNetwork, self).__init__() 
        self.input_dim = input_dim
        self.factor = factor
        self.num_layers = num_layers

        output_dim = int(self.input_dim / factor)
        output_dim = self.input_dim

        self.activation = torch.nn.ReLU() 
        self.dropout = torch.nn.Dropout3d(p=dropout) 
        layers = []

        kernel_size = int(64/self.num_layers)

        for i in range(num_layers):
            layers.append(torch.nn.ConvTranspose3d(input_dim, output_dim, kernel_size, padding=0)) 
            layers.append(self.activation) 
            layers.append(self.dropout) 
            input_dim = output_dim
            output_dim = max(32, int(input_dim / factor)) 

        self.output_dim = output_dim
        # per pixel per class
        conv_last = torch.nn.ConvTranspose3d(output_dim, num_blocks, 1, padding=0)
        layers.append(conv_last) 
        self.layers = torch.nn.ModuleList(layers) 

    def forward(self, encoded):
        # encoded: [batch, seq_len, input_dim]
        bsz, input_dim = encoded.data.shape
        # reshape [bsz, 4, 4, input_dim/8]
        encoded = encoded.reshape((bsz, -1, 1, 1, 1))

        for layer in self.layers:
            print(f"layer {layer}") 
            encoded = layer(encoded) 
            print(f"encoded {encoded.shape}") 
        # output: [batch, width, height] 
        output = encoded.reshape((bsz, 20, 64, 64, 64))
        return output 
