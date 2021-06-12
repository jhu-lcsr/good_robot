import torch
from torch import nn as nn

from learning.inputs.common import empty_float_tensor, cuda_var
from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.dbg_writer import DebugWriter
from visualization import Presenter
from utils.simple_profiler import SimpleProfiler

PROFILE = False


class LeakyIntegratorGlobalMap(MapTransformerBase):

    def __init__(self, source_map_size, world_size_px, world_size_m, lamda=0.2):
        super(LeakyIntegratorGlobalMap, self).__init__(source_map_size, world_size_px, world_size_m)
        self.map_size_px = source_map_size
        self.world_size_px = world_size_px
        self.world_size_m = world_size_m
        self.child_transformer = MapTransformerBase(source_map_size, world_size_px, world_size_m)
        self.lamda = lamda

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.map_memory = []
        self.coverage_memory = []

        self.dbg_t = None
        self.seq = 0

    def init_weights(self):
        pass

    def reset(self):
        super(LeakyIntegratorGlobalMap, self).reset()
        self.map_memory = []
        self.coverage_memory = []
        self.child_transformer.reset()
        self.seq = 0

    def cuda(self, device=None):
        MapTransformerBase.cuda(self, device)
        self.child_transformer.cuda(device)
        return self

    def dbg_write_extra(self, map, pose):
        if DebugWriter().should_write():
            map = map[0:1, 0:3]
            self.seq += 1
            # Initialize a transformer module
            if pose is not None:
                if self.dbg_t is None:
                    self.dbg_t = MapTransformerBase(self.map_size_px, self.world_size_px, self.world_size_m).to(map.device)

                # Transform the prediction to the global frame and write out to disk.
                self.dbg_t.set_map(map, pose)
                map_global, _ = self.dbg_t.get_map(None)
            else:
                map_global = map
            DebugWriter().write_img(map_global[0], "gif_overlaid", args={"world_size": self.world_size_px, "name": "sm"})

    def forward(self, images_w, coverages_w, add_mask=None, reset_mask=None, show=False):
        #show="li"
        self.prof.tick(".")
        batch_size = len(images_w)

        assert add_mask is None or add_mask[0] is not None, "The first observation in a sequence needs to be used!"

        masked_observations_w_add = self.lamda * images_w * coverages_w

        all_maps_out_w = []
        all_coverages_out_w = []

        self.prof.tick("maps_to_global")

        # TODO: Draw past trajectory on an extra channel of the semantic map
        # Step 2: Integrate serially in the global frame
        for i in range(batch_size):
            if len(self.map_memory) == 0 or (reset_mask is not None and reset_mask[i]):
                new_map_w = images_w[i:i + 1]
                new_map_cov_w = coverages_w[i:i+1]

            # Allow masking of observations
            elif add_mask is None or add_mask[i]:
                # Get the current global-frame map
                map_g = self.map_memory[-1]
                map_cov_g = self.coverage_memory[-1]
                cov_w = coverages_w[i:i+1]
                obs_cov_g = masked_observations_w_add[i:i+1]

                # Add the observation into the map using a leaky integrator rule (TODO: Output lamda from model)
                new_map_cov_w = torch.clamp(map_cov_g + cov_w, 0, 1)
                new_map_w = (1 - self.lamda) * map_g + obs_cov_g + self.lamda * map_g * (1 - cov_w)
            else:
                new_map_w = self.map_memory[-1]
                new_map_cov_w = self.coverage_memory[-1]

            self.map_memory.append(new_map_w)
            self.coverage_memory.append(new_map_cov_w)
            all_maps_out_w.append(new_map_w)
            all_coverages_out_w.append(new_map_cov_w)

            #Presenter().show_image(new_map_cov_w.data[0, 0:3], "map_cov", torch=True, scale=8, waitkey=1)
            if show != "":
                Presenter().show_image(new_map_cov_w.data[0, 0:3], show, torch=True, scale=8, waitkey=1)

        self.prof.tick("integrate")

        # Step 3: Convert all maps to local frame
        all_maps_w = torch.cat(all_maps_out_w, dim=0)
        all_coverages_out_w = torch.cat(all_coverages_out_w, dim=0)

        # Write gifs for debugging
        #self.dbg_write_extra(all_maps_w, None)

        self.prof.tick("maps_to_local")
        self.prof.loop()
        self.prof.print_stats(10)

        return all_maps_w, all_coverages_out_w