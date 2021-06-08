import torch
from torch import nn as nn

from learning.inputs.common import empty_float_tensor, cuda_var
from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.dbg_writer import DebugWriter
from learning.inputs.partial_2d_distribution import Partial2DDistribution
from visualization import Presenter
from utils.simple_profiler import SimpleProfiler

PROFILE = False


class MapBatchFillMissing(MapTransformerBase):

    def __init__(self, source_map_size_px, world_size_px, world_size_m):
        super(MapBatchFillMissing, self).__init__(source_map_size_px, world_size_px, world_size_m)
        self.map_size = source_map_size_px
        self.world_size = world_size_px
        self.world_size_m = world_size_m
        self.child_transformer = MapTransformerBase(source_map_size_px, world_size_px, world_size_m)

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.map_memory = MapTransformerBase(source_map_size_px, world_size_px, world_size_m)

        self.last_observation = None

        self.dbg_t = None
        self.seq = 0

    def init_weights(self):
        pass

    def reset(self):
        super(MapBatchFillMissing, self).reset()
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

    def forward(self, select_dist, all_cam_poses, plan_mask=None, show=False):
        #show="li"
        self.prof.tick(".")

        # During rollout, plan_mask will alternate between [True] and [False]
        if plan_mask is None:
            all_dist = select_dist
            return all_dist, all_cam_poses

        full_batch_size = len(all_cam_poses)

        all_dists_out_r = []

        self.prof.tick("maps_to_global")

        # For each timestep, take the latest map that was available, transformed into this timestep
        # Do only a maximum of one transformation for any map to avoid cascading of errors!
        ptr = 0
        for i in range(full_batch_size):
            this_pose = all_cam_poses[i:i+1]
            if plan_mask[i]:
                this_obs = (select_dist[ptr:ptr + 1], this_pose)
                ptr += 1
                self.last_observation = this_obs
            else:
                assert self.last_observation is not None, "The first observation in a sequence needs to be used!"
                last_map, last_pose = self.last_observation

                # TODO: See if we can speed this up. Perhaps batch for all timesteps inbetween observations
                self.child_transformer.set_map(last_map.inner_distribution, last_pose)
                x = self.child_transformer.get_map(this_pose)
                this_obs = Partial2DDistribution(x, last_map.outer_prob_mass)

            all_dists_out_r.append(this_obs)

            if show != "":
                Presenter().show_image(this_obs.inner_distribution.data[0, 0:3], show, torch=True, scale=8, waitkey=50)

        self.prof.tick("integrate")

        inner_list = [x.inner_distribution for x in all_dists_out_r]
        outer_list = [x.outer_prob_mass for x in all_dists_out_r]

        all_dists_out_r = Partial2DDistribution(torch.cat(inner_list, dim=0), torch.cat(outer_list, dim=0))

        self.prof.tick("maps_to_local")
        self.prof.loop()
        self.prof.print_stats(10)

        return all_dists_out_r, all_cam_poses
