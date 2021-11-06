####################################################
# implement different levels of parameter sharing in UNet 
####################################################
import copy 
import torch 
from unet_module import UNetWithBlocks, UNetWithAttention, UNetWithLanguage, UNetNoNorm, UNetForBERT

SHARE_LEVELS = {"none": 0,
                "embed": 1,
                "encoder": 2,
                "unet": 3}


UNET_CLASSES = {"unet_with_blocks": UNetWithBlocks,
                "unet_with_attention": UNetWithAttention,
                "unet_with_langauge": UNetWithLanguage,
                "unet_no_norm": UNetNoNorm,
                "unet_for_bert": UNetForBERT} 



class SharedUNet(torch.nn.Module):
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
                depth: int = 7,
                unet_type: str = "unet_with_attention",
                share_level: str = "encoder",
                fusion: str = "concat",
                device: torch.device = "cpu",
                do_reconstruction: bool = False):
        super(SharedUNet, self).__init__()        

        self.share_level = SHARE_LEVELS[share_level]
        self.compute_block_dist = False
        self.unet_type = unet_type
        self.do_reconstruction = do_reconstruction 

        if self.share_level < 2:
            # need to create copy encoder 
            next_lang_encoder = copy.deepcopy(lang_encoder) 
            if self.share_level < 1:
                # need to create copy embedder too 
                next_lang_embedder = copy.deepcopy(lang_embedder) 
            else:
                next_lang_embedder = lang_embedder
        else:
            next_lang_encoder = lang_encoder 
            next_lang_embedder = lang_embedder

        prev_lang_embedder = copy.deepcopy(lang_embedder) 
        prev_lang_encoder = copy.deepcopy(lang_encoder) 
    
        # always define this one 
        encoder_cls = UNET_CLASSES[self.unet_type]
        self.next_encoder = encoder_cls(in_channels=in_channels,
                                       out_channels=out_channels,
                                       lang_embedder=next_lang_embedder,
                                       lang_encoder=next_lang_encoder,
                                       hc_large=hc_large,
                                       hc_small=hc_small,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       num_layers=num_layers,
                                       num_blocks=num_blocks,
                                       #mlp_num_layers=mlp_num_layers,
                                       dropout=dropout,
                                       depth=depth,
                                       device=device,
                                       do_reconstruction=do_reconstruction) 

        if self.share_level < 3: 
            # make a new module if not shared 
            self.prev_encoder = encoder_cls(in_channels=in_channels,
                                           out_channels=out_channels,
                                           lang_embedder=prev_lang_embedder,
                                           lang_encoder=prev_lang_encoder,
                                           hc_large=hc_large,
                                           hc_small=hc_small,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           num_layers=num_layers,
                                           num_blocks=num_blocks,
                                           #mlp_num_layers=mlp_num_layers,
                                           dropout=dropout,
                                           depth=depth,
                                           device=device,
                                           do_reconstruction=do_reconstruction)
        else: 
            # make a pointer 
            self.prev_encoder = self.next_encoder

    def forward(self, data_batch):
        prev_output = self.prev_encoder(data_batch) 
        next_output = self.next_encoder(data_batch) 

        return (next_output, prev_output) 



