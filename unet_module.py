from collections import deque
import pdb 

import torch
import torch.nn.functional as F

from image_encoder import FinalClassificationLayer
from mlp import MLP 


class BaseUNet(torch.nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                hc_large: int,
                hc_small: int, 
                kernel_size: int = 5,
                stride: int = 2,
                num_layers: int = 5,
                num_blocks: int = 20,
                device: torch.device = "cpu"):
        super(BaseUNet, self).__init__()

        # placeholders 
        self.compute_block_dist = False 

        self.device = device

        self.num_layers = num_layers
        self.num_blocks = num_blocks

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

        self.final_layer = FinalClassificationLayer(int(out_channels/4), out_channels, self.num_blocks + 1) 

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
                num_blocks: int = 20,
                device: torch.device = "cpu"):
        super(UNetWithLanguage, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               hc_large=hc_large,
                                               hc_small=hc_small,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               num_layers=num_layers,
                                               num_blocks=num_blocks,
                                               device=device)

        pad = int(kernel_size / 2) 

        self.lang_embedder = lang_embedder
        self.lang_encoder = lang_encoder
        self.lang_embedder.set_device(self.device) 
        self.lang_encoder.set_device(self.device) 

        self.lang_projections = [] 
        for i in range(self.num_layers):
            lang_proj = torch.nn.Linear(self.lang_encoder.output_size, hc_large) 
            self.lang_projections.append(lang_proj)
        self.lang_projections = torch.nn.ModuleList(self.lang_projections) 
        self.lang_projections = self.lang_projections.to(self.device) 

        self.upconv_modules = torch.nn.ModuleList() 

        # need extra dims for concating language 
        first_upconv = torch.nn.ConvTranspose2d(2*hc_large, hc_large, kernel_size, stride=stride, padding=pad)
        self.upconv_modules.append(first_upconv)

        for i in range(num_layers-3): 
            upconv = torch.nn.ConvTranspose2d(3*hc_large, hc_large, kernel_size, stride=stride, padding = pad)
            self.upconv_modules.append(upconv) 

        penult_upconv = torch.nn.ConvTranspose2d(3*hc_large, hc_small, kernel_size, stride=stride, padding=pad) 
        self.upconv_modules.append(penult_upconv) 
        final_upconv = torch.nn.ConvTranspose2d(2*hc_large + hc_small, out_channels, kernel_size, stride=stride, padding=pad) 
        self.upconv_modules.append(final_upconv) 

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
        image_input = image_input.to(self.device) 
        # store downconv results in stack 
        downconv_results = deque() 
        lang_results = deque() 
        downconv_sizes = deque() 
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

            # get language projection at that layer 
            lang_proj = self.lang_projections[i]
            lang = lang_proj(sent_encoding)
            # expand language for tiling 
            bsz, __, width, height = out.shape
            lang = lang.view((bsz, -1, 1, 1))
            lang = lang.repeat((1, 1, width, height)) 
            lang_results.append(lang) 
            # concat language in 
            downconv_sizes.append(out.size())
            out_with_lang = torch.cat([out, lang], 1)
            downconv_results.append(out_with_lang) 
            if i == self.num_layers-1:
                # at end set out include lang
                out = out_with_lang
            

        # pop off last one 
        downconv_sizes.pop() 
        downconv_results.pop() 
        
        # go back up the U, concatenating residuals and language 
        for i in range(self.num_layers): 
            # concat the corresponding side of the U
            upconv = self.upconv_modules[i]
            if i > 0:
                resid_data = downconv_results.pop() 
                out = torch.cat([resid_data, out], 1)
            if i < self.num_layers-1:
                desired_size = downconv_sizes.pop() 
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


class UNetWithBlocks(UNetWithLanguage):
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
                num_blocks: int = 20,
                mlp_num_layers: int = 3, 
                dropout: float = 0.20,
                device: torch.device = "cpu"):
        super(UNetWithBlocks, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               lang_embedder=lang_embedder,
                                               lang_encoder=lang_encoder,
                                               hc_large=hc_large,
                                               hc_small=hc_small,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               num_layers=num_layers,
                                               num_blocks=num_blocks,
                                               device=device)

        self.compute_block_dist = True 
        # TODO (elias): automatically infer this size when the num_layers is different 
        width = int(64**(1/(num_layers-1))) 
        self.block_prediction_module = MLP(input_dim  = 2*width*width*hc_large,
                                           hidden_dim = 2*hc_large,
                                           output_dim = num_blocks + 1,
                                           num_layers = mlp_num_layers, 
                                           dropout = dropout) 

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
        image_input = image_input.to(self.device) 
        # store downconv results in stack 
        downconv_results = deque() 
        lang_results = deque() 
        downconv_sizes = deque() 
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

            # get language projection at that layer 
            lang_proj = self.lang_projections[i]
            lang = lang_proj(sent_encoding)
            # expand language for tiling 
            bsz, __, width, height = out.shape
            lang = lang.view((bsz, -1, 1, 1))
            lang = lang.repeat((1, 1, width, height)) 
            lang_results.append(lang) 
            # concat language in 
            downconv_sizes.append(out.size())
            out_with_lang = torch.cat([out, lang], 1)
            downconv_results.append(out_with_lang) 
            if i == self.num_layers-1:
                # at end set out include lang
                out = out_with_lang
            
        # predict blocks from deepest downconv 
        out_for_blocks = out.view((bsz, -1))
        pred_block_logits = self.block_prediction_module(out_for_blocks) 

        # pop off last one 
        downconv_sizes.pop() 
        downconv_results.pop() 
        
        # go back up the U, concatenating residuals and language 
        for i in range(self.num_layers): 
            # concat the corresponding side of the U
            upconv = self.upconv_modules[i]
            if i > 0:
                resid_data = downconv_results.pop() 
                out = torch.cat([resid_data, out], 1)
            if i < self.num_layers-1:
                desired_size = downconv_sizes.pop() 
            else:
                desired_size = image_input.size() 

            out = self.activation(upconv(out, output_size = desired_size))
                
            # last layer has no norm 
            if i < self.num_layers: 
                upnorm = self.upnorms[i-1]
                out = upnorm(out)

        out = self.final_layer(out) 

        to_ret = {"next_position": out,
                 "pred_block_logits": pred_block_logits}

        return to_ret 
