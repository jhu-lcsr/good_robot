from collections import deque
import pdb 

import torch
import torch.nn.functional as F

from image_encoder import FinalClassificationLayer


class BaseUNet(torch.nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                hc_large: int,
                hc_small: int, 
                kernel_size: int = 5,
                stride: int = 2,
                num_layers: int = 5,
                device: torch.device = "cpu"):
        super(BaseUNet, self).__init__()

        # placeholders 
        self.compute_block_dist = False 

        self.device = device

        self.num_layers = num_layers

        self.downconv_modules = []
        self.upconv_modules = []
        self.upconv_results = []

        self.downnorms = []
        self.upnorms = []
            
        self.hc_large = hc_large
        self.hc_small = hc_small

        pad = int(kernel_size / 2) 

        self.activation = torch.nn.ReLU()

        # exception at first layer for shape  
        first_downconv = torch.nn.Conv2d(in_channels, hc_large, kernel_size, stride=stride, padding=pad)
        first_upconv = torch.nn.ConvTranspose2d(hc_large, hc_large, kernel_size, stride=stride, padding=pad)
        first_downnorm = torch.nn.InstanceNorm2d(hc_large)
        first_upnorm   = torch.nn.InstanceNorm2d(hc_large)  

        self.downconv_modules.append(first_downconv) 
        self.upconv_modules.append(first_upconv)
        self.downnorms.append(first_downnorm) 
        self.upnorms.append(first_upnorm) 

        for i in range(num_layers-3): 
            downconv = torch.nn.Conv2d(hc_large, hc_large, kernel_size, stride=stride, padding=pad)
            downnorm = torch.nn.InstanceNorm2d(hc_large) 
            upconv = torch.nn.ConvTranspose2d(2*hc_large, hc_large, kernel_size, stride=stride, padding = pad)
            upnorm = torch.nn.InstanceNorm2d(hc_large) 

            self.downconv_modules.append(downconv) 
            self.upconv_modules.append(upconv) 
            self.downnorms.append(downnorm) 
            self.upnorms.append(upnorm) 

        penult_downconv = torch.nn.Conv2d(hc_large, hc_large, kernel_size, stride=stride, padding=pad)  
        penult_downnorm = torch.nn.InstanceNorm2d(hc_large) 
        penult_upconv = torch.nn.ConvTranspose2d(2*hc_large, hc_small, kernel_size, stride=stride, padding=pad) 
        penult_upnorm = torch.nn.InstanceNorm2d(hc_small) 

        self.downconv_modules.append(penult_downconv) 
        self.upconv_modules.append(penult_upconv) 
        self.downnorms.append(penult_downnorm) 
        self.upnorms.append(penult_upnorm) 
        
        final_downconv = torch.nn.Conv2d(hc_large, hc_large, kernel_size, stride=stride, padding=pad) 
        final_upconv = torch.nn.ConvTranspose2d(hc_large + hc_small, out_channels, kernel_size, stride=stride, padding=pad) 

        self.downconv_modules.append(final_downconv)
        self.upconv_modules.append(final_upconv) 

        self.downconv_modules = torch.nn.ModuleList(self.downconv_modules)
        self.upconv_modules = torch.nn.ModuleList(self.upconv_modules)
        self.downnorms = torch.nn.ModuleList(self.downnorms) 
        self.upnorms = torch.nn.ModuleList(self.upnorms) 

        self.final_layer = FinalClassificationLayer(int(out_channels/4), out_channels, 21) 

        self.downconv_modules = self.downconv_modules.to(self.device)
        self.upconv_modules = self.upconv_modules.to(self.device)
        self.downnorms = self.downnorms.to(self.device) 
        self.upnorms = self.upnorms.to(self.device)
        self.final_layer = self.final_layer.to(self.device) 
        self.activation = self.activation.to(self.device) 
        
    def forward(self, input_dict):
        image_input = input_dict["previous_position"]
        # store downconv results in stack 
        downconv_results = deque() 
        # start with image input 
        out = image_input     

        # get down outputs, going down U
        for i in range(self.num_layers): 
            downconv = self.downconv_modules[i]
            out = self.activation(downconv(out)) 
            # last layer has no norm 
            if i < self.num_layers-1: 
                downnorm = self.downnorms[i-1]
                out = downnorm(out) 
                downconv_results.append(out) 

        # go back up the U, concatenating residuals back in 
        for i in range(self.num_layers): 
            # concat the corresponding side of the U
            upconv = self.upconv_modules[i]
            if i > 0:
                resid_data = downconv_results.pop() 
                out = torch.cat([resid_data, out], 1)
            
            if i < self.num_layers-1:
                desired_size = downconv_results[-1].size()
            else:
                desired_size = image_input.size() 

            out = self.activation(upconv(out, output_size = desired_size))
                
            # last layer has no norm 
            if i < self.num_layers: 
                upnorm = self.upnorms[i-1]
                out = upnorm(out)

        out = self.final_layer(out) 

        to_ret = {"next_position": out,
                 "pred_block_logits": None}
        return to_ret 
        
class UNetWithLanguage(BaseUNet):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                lang_embedder: torch.nn.Module,
                lang_encoder: torch.nn.Module,
                hc_large: int,
                hc_small: int, 
                kernel_size: int = 5,
                stride: int = 2,
                num_layers: int = 5,
                device: torch.device = "cpu"):
        super(UNetWithLanguage, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               hc_large=hc_large,
                                               hc_small=hc_small,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               num_layers=num_layers,
                                               device=device)

        self.lang_embedder = lang_embedder
        self.lang_encoder = lang_encoder
        self.lang_embedder.set_device(self.device) 
        self.lang_encoder.set_device(self.device) 

        self.lang_projections = [] 
        for i in range(self.num_layers):
            lang_proj = torch.nn.Linear(self.lang_encoder.output_size, hc_large * hc_large) 
            self.lang_projections.append(lang_proj)
        self.lang_projections = torch.nn.ModuleList(self.lang_projections) 
        self.lang_projections = self.lang_projections.to(self.device) 

            
    def forward(self, data_batch):
        lang_input = data_batch["command"]
        lang_length = data_batch["length"]
        # sort lengths 
        lengths = data_batch["length"]
        lengths = [(i,x) for i, x in enumerate(lengths)]
        lengths = sorted(lengths, key = lambda x: x[1], reverse=True)
        idxs, lengths = zip(*lengths) 
        # tensorize lengths 
        lengths = torch.tensor(lengths).float() 
        lengths = lengths.to(self.device) 

        # embed langauge 
        lang_embedded = torch.cat([self.lang_embedder(lang_input[i]).unsqueeze(0) for i in idxs], 
                                    dim=0).to(self.device)

        # encode
        lang_output = self.lang_encoder(lang_embedded, lengths) 
        
        # get language output as sentence embedding 
        sent_encoding = lang_output["sentence_encoding"] 
    
        image_input = data_batch["previous_position"]
        # store downconv results in stack 
        downconv_results = deque() 
        lang_results = deque() 
        # start with image input 
        out = image_input     

        # get down outputs, going down U
        for i in range(self.num_layers): 
            # do language first 
            lang_proj = self.lang_projections[i]
            lang = F.normalize(lang_proj(sent_encoding)).reshape((-1, self.hc_large, self.hc_large, 1, 1))
            lang_results.append(lang) 

            downconv = self.downconv_modules[i]
            out = self.activation(downconv(out)) 
            # last layer has no norm 
            if i < self.num_layers-1: 
                downnorm = self.downnorms[i-1]
                out = downnorm(out) 
                # convolve by language after norm 
                out = F.conv2d(out, lang) 
                downconv_results.append(out) 

            else:
                # convolve by language anyway 
                out = F.conv2d(out, lang) 
        


        # go back up the U, concatenating residuals and language 
        for i in range(self.num_layers): 
            # concat the corresponding side of the U
            upconv = self.upconv_modules[i]
            if i > 0:
                resid_data = downconv_results.pop() 
                out = torch.cat([resid_data, out], 1)
            
            if i < self.num_layers-1:
                desired_size = downconv_results[-1].size()
            else:
                desired_size = image_input.size() 

            out = self.activation(upconv(out, output_size = desired_size))
                
            # last layer has no norm 
            if i < self.num_layers: 
                upnorm = self.upnorms[i-1]
                out = upnorm(out)

        out = self.final_layer(out) 

        to_ret = {"next_position": out,
                 "pred_block_logits": None}
        return to_ret 


