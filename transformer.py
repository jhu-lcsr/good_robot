import math
import torch
from torch import nn 
from torch.nn import functional as F
import einops 
from einops import rearrange, repeat 
from allennlp.nn.util import get_range_vector, get_device_of, add_positional_features

import pdb 

torch.manual_seed(12) 

# based on https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
def add_positional_features_2d(
    tensor: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4
):
    _, timesteps, hidden_dim = tensor.size()

    x_timesteps = int(math.sqrt(timesteps))
    y_timesteps = x_timesteps

    # put back into row-column format 
    tensor = rearrange(tensor, 'b (x y) h -> b x y h', x = x_timesteps, y = y_timesteps)


    timestep_range = get_range_vector(x_timesteps, get_device_of(tensor)).data.float()
    # We're generating both cos and sin frequencies,
    # so half for each.
    num_timescales = hidden_dim // 2
    timescale_range = get_range_vector(num_timescales, get_device_of(tensor)).data.float()

    log_timescale_increments = math.log(float(max_timescale) / float(min_timescale)) / float(
        num_timescales - 1
    )
    inverse_timescales = min_timescale * torch.exp(timescale_range * -log_timescale_increments)

    # Broadcasted multiplication - shape (timesteps, num_timescales)
    scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
    # shape (timesteps, 2 * num_timescales)
    sinusoids = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
    if hidden_dim % 2 != 0:
        # if the number of dimensions is odd, the cos and sin
        # timescales had size (hidden_dim - 1) / 2, so we need
        # to add a row of zeros to make up the difference.
        sinusoids = torch.cat([sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)

    sinusoids = sinusoids.unsqueeze(0)
    sinusoids = repeat(sinusoids, '() n d -> m n d', m=y_timesteps )
    tensor = tensor + sinusoids.unsqueeze(0)
    tensor = rearrange(tensor, 'b x y h -> b (x y) h')

    return tensor

def image_to_tiles(image, tile_size):
    tiler = torch.nn.Unfold(kernel_size = tile_size, stride = tile_size)
    output = tiler(image)
    output = rearrange(output, 'b n c -> b c n') 
    return output 


#def image_to_tiles(image, tile_size):
#    """tiles an image into image/tile_size tile_size x tile_size image tiles
#    image: torch.Tensor
#        [batch, channels, width, height]
#    tile_size: int
#    """
#    try:
#        assert(image.shape[-1] % tile_size == 0)
#    except AssertionError:
#        raise AssertionError(f"image width and height {image.shape[-1]} must be divisible by tile_size {tile_size}") 
#
#    p = tile_size 
#    # from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py
#    new_image = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)            
#    return new_image 

def upsample_tiles(tiled_output, tile_size):
    """
    takes tiled output, which is a distribution per tile, and upsamples it to a per-pixel distribution.
    All pixels corresponding to one tile get the same value (the tile value)
    """
    tiled_output = tiled_output.squeeze(-1)
    # rearrange so that the support of the dist is the channel dimension 
    tiled_output = rearrange(tiled_output, 'b n c -> b c n')
    # n here is the number of tiles, i.e. the unrolled image 
    b, c, n = tiled_output.shape 

    # get total number of pixels in the image 
    image_size = n * tile_size**2
    # take the sqrt to get the image width 
    w = int(math.sqrt(image_size))
    # this list will contain each tile, which we'll later concatenate together  
    output_image = [[None for j in range(w // tile_size)] for i in range(w//tile_size)]

    # iterate over valid row and column indices 
    for row_idx in range(int(w / tile_size)):
        for col_idx in range(int(w / tile_size)):
            # get the tile index 
            abs_idx = (row_idx * (w // tile_size)) + col_idx 
            # get the values of the tile at that index
            tile_values = tiled_output[:,:,abs_idx].unsqueeze(-1)
            # repeat those for a pxp square in the image 
            # [b, c, p^2]
            tile = tile_values.repeat((1, 1, tile_size**2))
            # [b, c, p, p]
            tile = tile.reshape((b, c, tile_size, tile_size))
            output_image[row_idx][col_idx] = tile

    # concatenate all the tiles in a row
    output_rows = [torch.cat(row, dim=3) for row in output_image]
    # concatenate all the rows into a final image 
    output = torch.cat(output_rows, dim=2) 
    return output 

def tiles_to_image(tile_output, tile_size, output_type = "per-pixel", upsample = True):
    """takes tiled output of a model and converts it back to an image-like shape
    tile_output: torch.Tensor
        [batch, num_tiles, tile_size^2 * channels] 
    tile_size: int
    """
    p = tile_size 
    if output_type == "per-pixel":
        image = tile_output 
        image = rearrange(image, 'b n (p1 p2 c) -> b c (n p1 p2)', p1 = p, p2 = p)
    elif output_type == "per-patch" and upsample: 
        # each of the n patches gets turned into a pxp image region 
        image = upsample_tiles(tile_output, tile_size) 
        return image 
    elif output_type == "patch-softmax" and upsample:
        # each of the n patches gets turned into a pxp image region 
        # take only the total max region 
        max_tile = torch.argmax(tile_output, dim = 1)
        # replicate old format 
        bsz, n, __, __ = tile_output.shape 
        tile_new_output = torch.zeros((bsz, n, 2, 1)).to(tile_output.device) 
        # set all to inf
        tile_new_output[:,:,0,:] = 1e8
        # except true tile 
        tile_new_output[:,max_tile, 1, :] = 1e8
        tile_new_output[:,max_tile, 0, :] = 0

        image = upsample_tiles(tile_new_output, tile_size) 
        return image 
    else:
        return tile_output 

    w = int(math.sqrt(image.shape[-1]))
    # instead of having a flat sequence of tiles, return a wxw square of tiles 
    image = rearrange(image, 'b c (w1 w2) -> b c w1 w2', w1 = w, w2 = w) 
    return image 

def _get_std_from_tensor(init_scale, tensor):
    if len(tensor.shape) > 2:
        in_d1, in_d2, out_d = tensor.shape
        in_d = in_d1 * in_d2
    else:
        in_d, out_d = tensor.shape

    # use gain to scale as in SmallInit of https://arxiv.org/pdf/1910.05895.pdf
    return (2 / (in_d + init_scale * out_d)) ** 0.5       


# from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., init_scale=4):
        super().__init__()

        self.init_scale = init_scale 

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self._init_weights() 

    def _init_weights(self): 
        for m in self.net:
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean = 0, std = _get_std_from_tensor(self.init_scale, m.weight))
                torch.nn.init.constant_(m.bias,  0) 

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., init_scale=4):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.init_scale = init_scale 

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self._init_weights() 

        self.attn_weight = None

    def _init_weights(self): 
        torch.nn.init.normal_(self.to_qkv.weight, mean = 0, std = _get_std_from_tensor(self.init_scale, self.to_qkv.weight))
        torch.nn.init.normal_(self.to_out[0].weight, mean = 0, std = _get_std_from_tensor(self.init_scale, self.to_out[0].weight))
        torch.nn.init.constant_(self.to_out[0].bias, 0) 

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            #mask = F.pad(mask.flatten(1), (1, 0), value = True)
            mask = mask.unsqueeze(1) 
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        self.attn_weight = attn 

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, init_scale, log_weights = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout, init_scale = init_scale))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout, init_scale = init_scale)))
            ]))

        self.log_weights = log_weights

    def forward(self, x, mask = None):
        attn_values = []
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
            if self.log_weights:
                attn_values.append(attn._modules["fn"]._modules["fn"].attn_weight.detach().cpu().numpy().tolist())
        return x, attn_values

class TransformerEncoder(torch.nn.Module):
    def __init__(self,
                 image_size: int, 
                 patch_size: int, 
                 language_embedder: torch.nn.Module, 
                 n_layers_shared: int,
                 n_layers_split: int,
                 n_classes: int = 2, 
                 channels: int = 21, 
                 n_heads: int = 8,
                 hidden_dim: int = 512,
                 ff_dim: int = 1024,
                 init_scale: int = 4,
                 dropout: float = 0.33,
                 embed_dropout: float = 0.33,
                 output_type: str = "per-pixel",
                 positional_encoding_type: str = "learned",
                 device: torch.device = "cpu",
                 log_weights: bool = False,
                 do_regression: bool = False,
                 do_reconstruction: bool = False,
                 long_command: bool = False,
                 pretrained_weights: str = None):
        super(TransformerEncoder, self).__init__() 

        self.compute_block_dist = False 
        self.output_type = output_type 
        self.init_scale = init_scale
        self.positional_encoding_type = positional_encoding_type
        self.do_regression = do_regression
        self.do_reconstruction = do_reconstruction

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.device = device 
        self.language_embedder = language_embedder 
        self.language_embedder.set_device(device)
        self.language_dim = self.language_embedder.output_dim 
    
        self.patch_size = patch_size

        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.patch_projection = torch.nn.Linear(patch_dim, hidden_dim)
        self.language_projection = torch.nn.Linear(self.language_dim, hidden_dim)
        self.long_command = long_command 

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.sep_token = torch.nn.Parameter(torch.randn(1, 1, hidden_dim))

        # first half of stack is dedicated to joint modeling, 2nd half splits previous and next 
        self.start_transformer = Transformer(hidden_dim, n_layers_shared, n_heads, ff_dim, dropout, init_scale, log_weights) 
        self.prev_transformer = Transformer(hidden_dim, n_layers_split, n_heads, ff_dim, dropout, init_scale, log_weights) 
        self.next_transformer = Transformer(hidden_dim, n_layers_split, n_heads, ff_dim, dropout, init_scale, log_weights) 

        self.dropout = torch.nn.Dropout(embed_dropout) 
        
        if output_type == "per-pixel":
            self.output_dim = self.patch_size**2 * n_classes
        elif output_type == "patch-softmax":
            self.output_dim = 1
        else:
            self.output_dim = n_classes 


        self.next_mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.output_dim)
        )
        self.prev_mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.output_dim) 
        )
        if self.do_regression:
            self.regression_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 3)
            )

        if self.do_reconstruction:
            # TODO (elias) maybe share these 
            self.prev_patch_class_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 21)
            )
            self.next_patch_class_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 21)
            )

        if self.long_command:
            self.cls_position = None
            self.source_color_class_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 8)
            )

        if pretrained_weights is not None:
            # initialize from pretrained 
            state_dict = torch.load(pretrained_weights)
            # remove vocabulary 
            new_state_dict = {k:v for k,v in state_dict.items() if "language_embedder.embeddings" not in k}
            self.load_state_dict(new_state_dict, strict=False)

    def get_lang_mask(self, lang): 
        bsz = len(lang)
        seq_len = len(lang[0])
        mask = torch.ones((bsz, seq_len))
        for i, seq in enumerate(lang):
            for j, word in enumerate(seq):
                if word == "<PAD>": 
                    mask[i,j] = 0

        return mask.bool().to(self.device) 

    def _prepare_input(self, image, language, mask = None):
        # patchify 
        p = self.patch_size 
        image_input = image_to_tiles(image, p).to(self.device) 
        batch_size, num_patches, __ = image_input.shape

        # get mask
        if mask is not None:
            # all image regions allowed 
            long_mask = torch.ones((batch_size, num_patches + 1)).bool().to(self.device)
            # concat in language mask 
            long_mask = torch.cat((long_mask, mask), dim = 1) 
        else:
            long_mask = None 

        # project and positionally encode image 
        model_input = self.patch_projection(image_input) 

        # project and positionally encode language 
        language_input = torch.cat([self.language_embedder(x).unsqueeze(0) for x  in language], dim = 0) 
        language_input = self.language_projection(language_input) 

        if self.long_command:
            self.cls_position = num_patches
        if self.positional_encoding_type == "learned":
            # add positional features to image patches 
            model_input += self.pos_embedding[:, :num_patches]
            language_input = add_positional_features(language_input) 
            # repeat [SEP] across batch 
            sep_tokens = repeat(self.sep_token, '() n d -> b n d', b=batch_size)
            # tack on [SEP]
            model_input = torch.cat([model_input, sep_tokens], dim = 1)
            # tack on language after [SEP]
            model_input = torch.cat([model_input, language_input], dim = 1)
        elif self.positional_encoding_type == "fixed": 
            # repeat [SEP] across batch 
            sep_tokens = repeat(self.sep_token, '() n d -> b n d', b=batch_size)
            # tack on [SEP]
            model_input = torch.cat([model_input, sep_tokens], dim = 1)
            # tack on language after [SEP]
            model_input = torch.cat([model_input, language_input], dim = 1)
            # add 1d positional to everying 
            model_input = add_positional_features(model_input) 
        elif self.positional_encoding_type == "fixed-separate": 
            # add fixed pos encoding to language and image separately 
            model_input = add_positional_features(model_input)
            language_input = add_positional_features(language_input)
            # repeat [SEP] across batch 
            sep_tokens = repeat(self.sep_token, '() n d -> b n d', b=batch_size)
            # tack on [SEP]
            model_input = torch.cat([model_input, sep_tokens], dim = 1)
            # tack on language after [SEP]
            model_input = torch.cat([model_input, language_input], dim = 1)
        elif self.positional_encoding_type == "fixed-2d-separate":
             # add fixed pos encoding to language and image separately 
            model_input = add_positional_features_2d(model_input)
            language_input = add_positional_features(language_input)
            # repeat [SEP] across batch 
            sep_tokens = repeat(self.sep_token, '() n d -> b n d', b=batch_size)
            # tack on [SEP]
            model_input = torch.cat([model_input, sep_tokens], dim = 1)
            # tack on language after [SEP]
            model_input = torch.cat([model_input, language_input], dim = 1)
        else:
            raise AssertionError(f"invalid positional type {self.positional_encoding_type}")

        return model_input, long_mask, num_patches

    def forward(self, batch_instance): 
        language = batch_instance['command']
        image = batch_instance['prev_pos_input']
        # TODO (elias): remove
        if False:
            pdb.set_trace() 
            import cv2 
            input = batch_instance['prev_pos_input'][0]
            #input = input[0:3,:,:]
            output = batch_instance["prev_pos_for_pred"][0]
            output = output.reshape((1, 224, 224)).repeat((3, 1, 1))
            output[output!=0]=255
            image_to_write = input + output 
            image_to_write = image_to_write.permute(1,2,0)
            path = "/home/estengel/scratch/fixed.png"
            cv2.imwrite(path, image_to_write.detach().cpu().numpy())
            img_alone = input.permute(1,2,0)
            path = "/home/estengel/scratch/fixed_img.png"
            cv2.imwrite(path, img_alone.detach().cpu().numpy())

            #input = batch_instance['prev_pos_input'][1]
            #depth_input = input[3:,:,:]
            #output = batch_instance["prev_pos_for_acc"][1]
            #output = output.reshape((1, 64, 64)).repeat((3, 1, 1))
            #output[output!=0]=255
            #depth_image = depth_input + output 
            #depth_image = depth_image.permute(1,2,0)
            #path = "/home/estengel/scratch/fixed_depth.png"
            #cv2.imwrite(path, depth_image.detach().cpu().numpy())

        language_mask = self.get_lang_mask(language) 
        tfmr_input, mask, n_patches = self._prepare_input(image, language, language_mask) 

        tfmr_output, __ = self.start_transformer(tfmr_input, mask) 
        prev_output, prev_attn_out = self.prev_transformer(tfmr_output, mask) 
        next_output, next_attn_out = self.next_transformer(tfmr_output, mask) 

        # trim off language 
        prev_just_image_output = prev_output[:, 0:n_patches, :]
        next_just_image_output = next_output[:, 0:n_patches, :]

        # run final MLP 
        prev_classes = self.prev_mlp_head(prev_just_image_output) 
        next_classes = self.next_mlp_head(next_just_image_output) 
        if self.do_regression: 
            # go off sep token
            next_pos_xyz = self.regression_head(next_output[:, n_patches, :]) 
        else:
            next_pos_xyz = None

        if self.do_reconstruction: 
            next_patch_class = self.next_patch_class_head(next_output[:,0:n_patches,:])
            prev_patch_class = self.prev_patch_class_head(prev_output[:,0:n_patches,:])
        else:
            next_patch_class, prev_patch_class = None, None


        # convert back to image 
        prev_image_output = tiles_to_image(prev_classes, self.patch_size, self.output_type, False).unsqueeze(-1) 
        next_image_output = tiles_to_image(next_classes, self.patch_size, self.output_type, False).unsqueeze(-1) 

        if self.long_command:
            sep_token = prev_output[:, self.cls_position, :]
            pred_source_color = self.source_color_class_head(sep_token)
        else:
            pred_source_color = None

        return {"prev_position": prev_image_output,
                "next_position": next_image_output,
                "next_per_patch_class": next_patch_class,
                "prev_per_patch_class": prev_patch_class,
                "next_pos_xyz": next_pos_xyz, 
                "pred_source_color": pred_source_color,
                "prev_attn_weights": prev_attn_out,
                "next_attn_weights": next_attn_out}

class ResidualTransformerEncoder(TransformerEncoder):
    def __init__(self,
                 image_size: int, 
                 patch_size: int, 
                 language_embedder: torch.nn.Module, 
                 n_layers_shared: int,
                 n_layers_split: int,
                 n_classes: int = 2, 
                 channels: int = 21, 
                 n_heads: int = 8,
                 hidden_dim: int = 512,
                 ff_dim: int = 1024,
                 init_scale: int = 4,
                 dropout: float = 0.33,
                 embed_dropout: float = 0.33,
                 output_type: str = "per-pixel",
                 positional_encoding_type: str = "learned",
                 device: torch.device = "cpu",
                 log_weights: bool = False,
                 do_regression: bool = False,
                 do_reconstruction: bool = False,
                 do_residual: bool = False):
        super(ResidualTransformerEncoder, self).__init__(image_size = image_size,
                                                         patch_size = patch_size,
                                                         language_embedder=language_embedder,
                                                         n_layers_shared=n_layers_shared,
                                                         n_layers_split = n_layers_split,
                                                         n_classes = n_classes,
                                                         channels = channels,
                                                         n_heads = n_heads,
                                                         hidden_dim = hidden_dim,
                                                         ff_dim = ff_dim,
                                                         init_scale = init_scale,
                                                         dropout = dropout, 
                                                         embed_dropout = embed_dropout,
                                                         output_type = output_type,
                                                         positional_encoding_type=positional_encoding_type,
                                                         device=device,
                                                         log_weights=log_weights,
                                                         do_regression=do_regression,
                                                         do_reconstruction=do_reconstruction)
        self.do_residual = do_residual
        if self.do_residual:
            self.next_transformer = Transformer(2*hidden_dim, n_layers_split, n_heads, ff_dim, dropout, init_scale, log_weights) 
            self.next_mlp_head = nn.Sequential(
                nn.LayerNorm(2*hidden_dim),
                nn.Linear(2*hidden_dim, self.output_dim)
            )
        else:
            self.next_transformer = Transformer(hidden_dim, n_layers_split, n_heads, ff_dim, dropout, init_scale, log_weights) 
            self.next_mlp_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.output_dim)
            )

    def forward(self, batch_instance): 
        language = batch_instance['command']
        image = batch_instance['prev_pos_input']

        language_mask = self.get_lang_mask(language) 
        tfmr_input, mask, n_patches = self._prepare_input(image, language, language_mask) 

        tfmr_output, __ = self.start_transformer(tfmr_input, mask) 
        prev_output, prev_attn_out = self.prev_transformer(tfmr_output, mask) 

        if self.do_residual:
            # residually connect shared output with prev output 
            next_input = torch.cat([prev_output, tfmr_output], dim = -1)
        else:
            # only stream through prev_output 
            next_input = prev_output 
        # CHANGE FROM TransformerEncoder: connect from prev_output to next
        next_output, next_attn_out = self.next_transformer(next_input, mask) 

        # trim off language 
        prev_just_image_output = prev_output[:, 0:n_patches, :]
        next_just_image_output = next_output[:, 0:n_patches, :]

        # run final MLP 
        prev_classes = self.prev_mlp_head(prev_just_image_output) 
        next_classes = self.next_mlp_head(next_just_image_output) 
        if self.do_regression: 
            # go off sep token
            next_pos_xyz = self.regression_head(next_output[:, n_patches, :]) 
        else:
            next_pos_xyz = None

        if self.do_reconstruction: 
            next_patch_class = self.next_patch_class_head(next_output[:,0:n_patches,:])
            prev_patch_class = self.prev_patch_class_head(prev_output[:,0:n_patches,:])
        else:
            next_patch_class, prev_patch_class = None, None


        # convert back to image 
        prev_image_output = tiles_to_image(prev_classes, self.patch_size, self.output_type, False).unsqueeze(-1) 
        next_image_output = tiles_to_image(next_classes, self.patch_size, self.output_type, False).unsqueeze(-1) 

        return {"prev_position": prev_image_output,
                "next_position": next_image_output,
                "next_per_patch_class": next_patch_class,
                "prev_per_patch_class": prev_patch_class,
                "next_pos_xyz": next_pos_xyz, 
                "prev_attn_weights": prev_attn_out,
                "next_attn_weights": next_attn_out}



