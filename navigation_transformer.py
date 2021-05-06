import math
import torch
from torch import nn 
from torch.nn import functional as F
import einops 
from einops import rearrange, repeat 
from allennlp.nn.util import get_range_vector, get_device_of, add_positional_features
from transformer import add_positional_features_2d, image_to_tiles, upsample_tiles, tiles_to_image, _get_std_from_tensor, Transformer, TransformerEncoder
import pdb 

torch.manual_seed(12) 

class NavigationTransformerEncoder(TransformerEncoder):
    def __init__(self,
                 image_size: int, 
                 patch_size: int, 
                 language_embedder: torch.nn.Module, 
                 n_layers: int,
                 n_classes: int = 2, 
                 channels: int = 3, 
                 n_heads: int = 8,
                 hidden_dim: int = 512,
                 ff_dim: int = 1024,
                 init_scale: int = 4,
                 dropout: float = 0.33,
                 embed_dropout: float = 0.33,
                 output_type: str = "per-pixel",
                 positional_encoding_type: str = "learned",
                 device: torch.device = "cpu",
                 locality_mask: bool = False, 
                 locality_neighborhood: int = 5, 
                 log_weights: bool = False):
        super(NavigationTransformerEncoder, self).__init__(image_size=image_size,
                                                           patch_size=patch_size,
                                                           language_embedder=language_embedder,
                                                           n_layers_shared=n_layers,
                                                           n_layers_split=0,
                                                           n_classes=n_classes,
                                                           channels=channels,
                                                           n_heads=n_heads,
                                                           hidden_dim=hidden_dim,
                                                           ff_dim=ff_dim,
                                                           init_scale=init_scale,
                                                           dropout=dropout,
                                                           embed_dropout=embed_dropout,
                                                           output_type=output_type,
                                                           positional_encoding_type=positional_encoding_type,
                                                           device=device)

        self.start_pos_projection = torch.nn.Linear(2, hidden_dim)
        self.locality_mask
        self.locality_neighborhood = locality_neighborhood

    def get_neighbors(self, patch_idx, num_patches, neighborhood = 5): 
        image_w = int(num_patches**(1/2))
        patch_idxs = np.arange(num_patches).reshape(image_w, image_w)
        patch_row = int(patch_idx / image_w) 
        patch_col = patch_idx % image_w 

        neighbor_idxs = patch_idxs[patch_row - neighborhood:patch_row + neighborhood, patch_col - neighborhood:patch_col + neighborhood]
        neighbor_idxs = neighbor_idxs.reshape(-1)
        return neighbor_idxs

    def get_image_local_mask(self, num_patches, image_dim, neighborhood = 5): 
        # make a mask so that each image patch can only attend to patches close to it 
        mask = torch.zeros((bsz, num_patches, num_patches))
        for i in range(num_patches):
            neighbors = self.get_neighbors(i, num_patches, neighborhood)
            mask[:,i,neighbors] = 1
        return mask.bool().to(self.device)

    def _prepare_input(self, image, language, start_pos, mask = None):
        # patchify 
        p = self.patch_size 
        image = image.permute(0,3,1,2).float() 
        image_input = image_to_tiles(image, p).to(self.device) 
        start_pos = start_pos.to(self.device)
        batch_size, num_patches, __ = image_input.shape

        # get mask
        if mask is not None:
            if self.locality_mask: 
                image_dim = int(num_patches**(1/2))
                long_mask = self.get_image_local_mask(num_patches, image_dim, neighborhood = self.locality_neighborhood)
                pdb.set_trace() 
            else:
                # all image regions allowed 
                long_mask = torch.ones((batch_size, num_patches + 1)).bool().to(self.device)
            # concat in language mask 
            long_mask = torch.cat((long_mask, mask), dim = 1) 
            # concat in one for positional token and sep 
            pos_token_mask = torch.ones((batch_size, 2)).bool().to(self.device)
            long_mask = torch.cat((long_mask, pos_token_mask), dim=1) 
        else:
            long_mask = None 

        # project and positionally encode image 
        model_input = self.patch_projection(image_input) 
        # project and positionally encode language 
        language_input = torch.cat([self.language_embedder(x).unsqueeze(0) for x  in language], dim = 0) 
        language_input = self.language_projection(language_input) 
        start_pos = self.start_pos_projection(start_pos).unsqueeze(1)

        if self.positional_encoding_type == "fixed-separate": 
            # add fixed pos encoding to language and image separately 
            model_input = add_positional_features(model_input)
            language_input = add_positional_features(language_input)
            # repeat [SEP] across batch 
            sep_tokens_1 = repeat(self.sep_token, '() n d -> b n d', b=batch_size)
            sep_tokens_2 = repeat(self.sep_token, '() n d -> b n d', b=batch_size)
            # tack on [SEP]
            model_input = torch.cat([model_input, sep_tokens_1], dim = 1)
            # tack on language after [SEP]
            model_input = torch.cat([model_input, language_input], dim = 1)
            # tack another [SEP]
            model_input = torch.cat([model_input, sep_tokens_2], dim = 1)
            # tack on start position 
            model_input = torch.cat([model_input, start_pos], dim = 1)
        else:
            raise AssertionError(f"invalid positional type {self.positional_encoding_type}")

        return model_input, long_mask, num_patches

    def forward(self, batch_instance): 
        language = batch_instance['command']
        image = batch_instance['input_image']
        start_pos = batch_instance['start_position'].float() 

        language_mask = self.get_lang_mask(language) 
        tfmr_input, mask, n_patches = self._prepare_input(image, language, start_pos, mask = language_mask) 

        tfmr_output, __ = self.start_transformer(tfmr_input, mask) 
        # trim off language 
        next_just_image_output = tfmr_output[:, 0:n_patches, :]
        # run final MLP 
        next_classes = self.next_mlp_head(next_just_image_output) 
        # convert back to image 
        next_image_output = tiles_to_image(next_classes, self.patch_size, self.output_type, False).unsqueeze(-1) 

        return {"next_position": next_image_output} 
