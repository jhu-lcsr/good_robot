import torch
from learning.datasets.aux_data_providers import get_top_down_ground_truth_dynamic_global
from learning.modules.map_transformer_base import MapTransformerBase
from learning.inputs.pose import get_noisy_poses_np, Pose
from utils.simple_profiler import SimpleProfiler

from visualization import Presenter

PROFILE = False


class MapPerturbation(MapTransformerBase):

    def __init__(self, source_map_size, world_size_px, world_size_m):#, pos_variance=0, rot_variance=0.5):
        super(MapPerturbation, self).__init__(source_map_size, world_size_px, world_size_m)
        self.map_size = source_map_size
        self.world_size_px = world_size_px
        self.world_size_m = world_size_m
        #self.pos_variance = pos_variance
        #self.rot_variance = rot_variance
        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)

    def init_weights(self):
        pass

    def cuda(self, device=None):
        MapTransformerBase.cuda(self, device)
        return self

    """
    def sample_noisy_poses(self, original_poses):
        noisy_poses = get_noisy_poses_np(original_poses.numpy(), self.pos_variance, self.rot_variance)
        noisy_poses = Pose(torch.from_numpy(noisy_poses.position), torch.from_numpy(noisy_poses.orientation))
        return noisy_poses
    """

    def show(self, perturbed_maps, unperturbed_maps, name):
        Presenter().show_image(unperturbed_maps.data[0], name + "_unperturbed", torch=True, waitkey=1, scale=4)
        Presenter().show_image(perturbed_maps.data[0], name + "_perturbed", torch=True, waitkey=1, scale=4)

    def forward(self, maps, map_poses_original, map_poses_w_noise, proc_mask=None, show=""):
        self.set_maps(maps, map_poses_original)
        # T#ODO: Remove this superfluous transformation
        #maps_g, _ = self.get_maps(None)
        #self.set_maps(maps_g, None)
        perturbed_maps, _ = self.get_maps(map_poses_w_noise)
        if show != "":
            self.show(perturbed_maps, maps, str(show))

        # WARNING: The actual poses are poses with noise. We're returning the original poses, so that other modules
        # don't rotate back and undo the random perturbations.
        # EDIT: Try outputting noisy poses instead to keep things consistent
        return perturbed_maps, map_poses_w_noise