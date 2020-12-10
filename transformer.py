import math
import torch
from torch import nn 
from torch.nn import functional as F
import einops 
from einops import rearrange, repeat 

import pdb 

# from https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)

# from https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
def get_device_of(tensor: torch.Tensor) -> int:
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()

# from https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
def add_positional_features(
    tensor: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4
):
    _, timesteps, hidden_dim = tensor.size()

    timestep_range = get_range_vector(timesteps, get_device_of(tensor)).data.float()
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
    return tensor + sinusoids.unsqueeze(0)

def image_to_tiles(image, tile_size):
    """tiles an image into image/tile_size tile_size x tile_size image tiles
    image: torch.Tensor
        [batch, channels, width, height]
    tile_size: int
    """
    try:
        assert(image.shape[-1] % tile_size == 0)
    except AssertionError:
        raise AssertionError(f"image width and height {image.shape[-1]} must be divisible by tile_size {tile_size}") 

    p = tile_size 
    # from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py
    new_image = einops.rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)            
    return new_image 

def tiles_to_image(tile_output, tile_size):
    """takes tiled output of a model and converts it back to an image-like shape
    tile_output: torch.Tensor
        [batch, num_tiles, tile_size^2 * channels] 
    tile_size: int
    """

    p = tile_size 
    image = einops.rearrange(tile_output, 'b n (p1 p2 c) -> b c (n p1 p2)', p1 = p, p2 = p)
    w = int(math.sqrt(image.shape[-1]))
    image = einops.rearrange(image, 'b c (w1 w2) -> b c w1 w2', w1 = w, w2 = w) 
    return image 

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
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

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

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class TransformerEncoder(torch.nn.Module):
    def __init__(self,
                 image_size: int, 
                 patch_size: int, 
                 language_embedder: torch.nn.Module, 
                 n_layers: int,
                 n_classes: int = 2, 
                 channels: int = 21, 
                 n_heads: int = 8,
                 hidden_dim: int = 512,
                 ff_dim: int = 1024,
                 dropout: float = 0.33,
                 embed_dropout: float = 0.33,
                 device: torch.device = "cpu"):
        super(TransformerEncoder, self).__init__() 

        self.compute_block_dist = False                                                                                                                         
        try:
            assert(n_layers % 2 == 0)
        except AssertionError:
            raise AssertionError("number of layers {n_layers} must be even") 

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

        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.sep_token = torch.nn.Parameter(torch.randn(1, 1, hidden_dim))

        # first half of stack is dedicated to joint modeling, 2nd half splits previous and next 
        self.start_transformer = Transformer(hidden_dim, int(n_layers/2), n_heads, ff_dim, dropout) 
        self.prev_transformer = Transformer(hidden_dim, int(n_layers/2), n_heads, ff_dim, dropout) 
        self.next_transformer = Transformer(hidden_dim, int(n_layers/2), n_heads, ff_dim, dropout) 

        self.dropout = torch.nn.Dropout(embed_dropout) 

        self.next_mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.patch_size**2 * n_classes)
        )
        self.prev_mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.patch_size**2 * n_classes)
        )

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
        image_input = image_to_tiles(image, p) 
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
        # add positional features to image patches 
        model_input += self.pos_embedding[:, :num_patches]

        # project and positionally encode language 
        language_input = torch.cat([self.language_embedder(x).unsqueeze(0) for x  in language], dim = 0) 
        language_input = self.language_projection(language_input) 
        language_input = add_positional_features(language_input) 

        # repeat [SEP] across batch 
        sep_tokens = repeat(self.sep_token, '() n d -> b n d', b=batch_size)
        # tack on [SEP]
        model_input = torch.cat([model_input, sep_tokens], dim = 1)
        # tack on language after [SEP]
        model_input = torch.cat([model_input, language_input], dim = 1)
        
        return model_input, long_mask, num_patches

    def forward(self, batch_instance): 
        language = batch_instance['command']
        image = batch_instance['prev_pos_input']
        language_mask = self.get_lang_mask(language) 

        tfmr_input, mask, n_patches = self._prepare_input(image, language, language_mask) 

        tfmr_output = self.start_transformer(tfmr_input, mask) 
        prev_output = self.prev_transformer(tfmr_output, mask) 
        next_output = self.next_transformer(tfmr_output, mask) 

        # trim off language 
        prev_just_image_output = prev_output[:, 0:n_patches, :]
        next_just_image_output = next_output[:, 0:n_patches, :]

        # run final MLP 
        prev_classes = self.prev_mlp_head(prev_just_image_output) 
        next_classes = self.next_mlp_head(next_just_image_output) 

        # convert back to image 
        prev_image_output = tiles_to_image(prev_classes, self.patch_size).unsqueeze(-1) 
        next_image_output = tiles_to_image(next_classes, self.patch_size).unsqueeze(-1) 

        return {"prev_position": prev_image_output,
                "next_position": next_image_output}



