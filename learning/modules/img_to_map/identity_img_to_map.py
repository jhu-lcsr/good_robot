from learning.modules.map_transformer_base import MapTransformerBase
from learning.models.semantic_map.grid_sampler import GridSampler

from utils.simple_profiler import SimpleProfiler

PROFILE = False


class IdentityImgToMap(MapTransformerBase):
    def __init__(self,
                 source_map_size, world_size_px,
                 world_size_px):
        super(IdentityImgToMap, self).__init__(source_map_size, world_size_px, world_size_m=world_size_px)

        self.grid_sampler = GridSampler()

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

    def cuda(self, device=None):
        MapTransformerBase.cuda(self, device)
        self.grid_sampler.cuda(device)

    def init_weights(self):
        pass

    def reset(self):
        super(IdentityImgToMap, self).reset()

    def forward(self, images, poses, sentence_embeds, parent=None, show=""):
        ...
        # TODO: Implement this

        return images, new_coverages