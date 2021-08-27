import torch
from torch import nn as nn
from torch.autograd import Variable

from learning.inputs.common import empty_float_tensor, cuda_var
from learning.modules.map_transformer_base import MapTransformerBase
from learning.modules.dbg_writer import DebugWriter
from visualization import Presenter
from utils.simple_profiler import SimpleProfiler
from transformations import poses_m_to_px

PROFILE = False


class DrawStartPosOnGlobalMap(MapTransformerBase):

    def __init__(self, source_map_size, world_size_px, world_size_m, lamda=0.2):
        super(DrawStartPosOnGlobalMap, self).__init__(source_map_size, world_size_px, world_size_m)
        self.map_size = source_map_size
        self.world_size_px = world_size_px
        self.world_size_m = world_size_m
        self.child_transformer = MapTransformerBase(source_map_size, world_size_px, world_size_m)

        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.start_pose = None
        self.last_emb = None

        self.dbg_t = None
        self.seq = 0

    def init_weights(self):
        pass

    def reset(self):
        super(DrawStartPosOnGlobalMap, self).reset()
        self.start_pose = None
        self.last_emb = None
        self.child_transformer.reset()
        self.seq = 0

    def cuda(self, device=None):
        MapTransformerBase.cuda(self, device)
        self.child_transformer.cuda(device)
        return self

    def get_start_poses(self, cam_poses_w, sentence_embeddings):
        # For each timestep, get the pose corresponding to the start of the instruction segment
        seq_len = len(sentence_embeddings)
        start_poses = []
        for i in range(seq_len):
            if self.last_emb is not None and (sentence_embeddings[i].data == self.last_emb).all():
                pass # Keep the same start pose since we're on the same segment
            else:
                self.last_emb = sentence_embeddings[i].data
                self.start_pose = cam_poses_w[i]
            start_poses.append(self.start_pose)
        return start_poses

    def forward(self, maps_w, sentence_embeddings, map_poses_w, cam_poses_w, show=False):
        #show="li
        self.prof.tick(".")
        batch_size = len(maps_w)

        # Initialize the layers of the same size as the maps, but with only one channel
        new_layer_size = list(maps_w.size())
        new_layer_size[1] = 1
        all_maps_out_w = empty_float_tensor(new_layer_size, self.is_cuda, self.cuda_device)

        start_poses = self.get_start_poses(cam_poses_w, sentence_embeddings)

        poses_img = [poses_m_to_px(as_pose, self.map_size, self.world_size_px, self.world_size_m) for as_pose in start_poses]
        #poses_img = poses_as_to_img(start_poses, self.world_size, batch_dim=True)

        for i in range(batch_size):
            x = min(max(int(poses_img[i].position.data[0]), 0), new_layer_size[2] - 1)
            y = min(max(int(poses_img[i].position.data[1]), 0), new_layer_size[2] - 1)
            all_maps_out_w[i, 0, x, y] = 10.0

        if show != "":
            Presenter().show_image(all_maps_out_w[0], show, torch=True, waitkey=1)

        self.prof.tick("draw")

        # Step 3: Convert all maps to local frame
        maps_out = torch.cat([Variable(all_maps_out_w), maps_w], dim=1)
        #all_maps_w = torch.cat(all_maps_out_w, dim=0)

        self.prof.loop()
        self.prof.print_stats(10)

        return maps_out, map_poses_w