import torch
from torch import nn as nn

from learning.inputs.common import empty_float_tensor, cuda_var
from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.dbg_writer import DebugWriter
from visualization import Presenter
from utils.simple_profiler import SimpleProfiler

PROFILE = False


class IdentityIntegratorMap(MapTransformerBase):

    def __init__(self, source_map_size, world_size_px, world_size_m):
        super(IdentityIntegratorMap, self).__init__(source_map_size, world_size_px, world_size_m)
        self.map_size = source_map_size
        self.world_size = world_size_px
        self.world_size_m = world_size_m
        self.child_transformer = MapTransformerBase(source_map_size, world_size_px, world_size_m)

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.map_memory = MapTransformerBase(source_map_size, world_size_px, world_size_m)

        self.last_observation = None

        self.dbg_t = None
        self.seq = 0

    def init_weights(self):
        pass

    def reset(self):
        super(IdentityIntegratorMap, self).reset()
        self.map_memory.reset()
        self.child_transformer.reset()
        self.seq = 0
        self.last_observation = None

    def cuda(self, device=None):
        MapTransformerBase.cuda(self, device)
        self.child_transformer.cuda(device)
        self.map_memory.cuda(device)
        return self

    def dbg_write_extra(self, map, pose):
        if DebugWriter().should_write():
            map = map[0:1, 0:3]
            self.seq += 1
            # Initialize a transformer module
            if pose is not None:
                if self.dbg_t is None:
                    self.dbg_t = MapTransformerBase(self.map_size, self.world_size, self.world_size_m).to(map.device)

                # Transform the prediction to the global frame and write out to disk.
                self.dbg_t.set_map(map, pose)
                map_global, _ = self.dbg_t.get_map(None)
            else:
                map_global = map
            DebugWriter().write_img(map_global[0], "gif_overlaid", args={"world_size": self.world_size, "name": "identity_integrator"})

    def forward(self, images, cam_poses, add_mask=None, show=False):
        #show="li"
        self.prof.tick(".")
        batch_size = len(cam_poses)

        assert add_mask is None or add_mask[0] is not None, "The first observation in a sequence needs to be used!"

        all_maps_out_r = []

        self.prof.tick("maps_to_global")

        # For each timestep, take the latest map that was available, transformed into this timestep
        # Do only a maximum of one transformation for any map to avoid cascading of errors!
        for i in range(batch_size):

            if add_mask is None or add_mask[i]:
                this_obs = (images[i:i+1], cam_poses[i:i+1])
                self.last_observation = this_obs
            else:
                last_obs = self.last_observation
                assert last_obs is not None, "The first observation in a sequence needs to be used!"

                self.child_transformer.set_map(last_obs[0], last_obs[1])
                this_obs = self.child_transformer.get_map(cam_poses[i:i+1])

            all_maps_out_r.append(this_obs[0])

            if show != "":
                Presenter().show_image(this_obs.data[0, 0:3], show, torch=True, scale=8, waitkey=50)

        self.prof.tick("integrate")

        # Step 3: Convert all maps to local frame
        all_maps_r = torch.cat(all_maps_out_r, dim=0)

        # Write gifs for debugging
        self.dbg_write_extra(all_maps_r, None)

        self.set_maps(all_maps_r, cam_poses)

        self.prof.tick("maps_to_local")
        self.prof.loop()
        self.prof.print_stats(10)

        return all_maps_r, cam_poses

    def forward_deprecated(self, images, cam_poses, add_mask=None, show=False):
        #show="li"
        self.prof.tick(".")
        batch_size = len(cam_poses)

        assert add_mask is None or add_mask[0] is not None, "The first observation in a sequence needs to be used!"

        # Step 1: All local maps to global:
        #  TODO: Allow inputing global maps when new projector is ready
        self.child_transformer.set_maps(images, cam_poses)
        observations_g, _ = self.child_transformer.get_maps(None)

        all_maps_out_g = []

        self.prof.tick("maps_to_global")

        # TODO: Draw past trajectory on an extra channel of the semantic map
        # Step 2: Integrate serially in the global frame
        for i in range(batch_size):

            # If we don't have a map yet, initialize the map to this observation
            if self.map_memory.latest_maps is None:
                self.map_memory.set_map(observations_g[i:i+1], None)

            # Allow masking of observations
            if add_mask is None or add_mask[i]:
                # Use the map from this frame
                map_g = observations_g[i:i+1]
                self.map_memory.set_map(map_g, None)
            else:
                # Use the latest available map oriented in global frame
                map_g, _ = self.map_memory.get_map(None)

            if show != "":
                Presenter().show_image(map_g.data[0, 0:3], show, torch=True, scale=8, waitkey=50)

            all_maps_out_g.append(map_g)

        self.prof.tick("integrate")

        # Step 3: Convert all maps to local frame
        all_maps_g = torch.cat(all_maps_out_g, dim=0)

        # Write gifs for debugging
        self.dbg_write_extra(all_maps_g, None)

        self.child_transformer.set_maps(all_maps_g, None)
        maps_r, _ = self.child_transformer.get_maps(cam_poses)
        self.set_maps(maps_r, cam_poses)

        self.prof.tick("maps_to_local")
        self.prof.loop()
        self.prof.print_stats(10)

        return maps_r, cam_poses