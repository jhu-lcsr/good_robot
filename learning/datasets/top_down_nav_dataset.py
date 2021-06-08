import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from utils.dict_tools import dict_slice

from deprecated.learning.datasets.sm_dataset_simple import SMSegmentDataset
from learning.datasets.top_down_dataset import TopDownDataset

from visualization import Presenter

IMG_WIDTH = 512
IMG_HEIGHT = 512

class TopDownNavDataset(Dataset):
    """
    Just like SMSegmentDataset, but returns images in top-down view instead of first-person view.
    """
    def __init__(self,
                 data=None,
                 env_list=None,
                 dataset_name="supervised",
                 seg_start_only=True,
                 aux_provider_names=[],
                 max_traj_len=None,
                 img_w=IMG_WIDTH,
                 img_h=IMG_HEIGHT,
                 map_w=None,
                 map_h=None):
        super(TopDownNavDataset, self).__init__()

        if map_w is None:
            map_w = map_w
            map_h = map_h

        self.segment_dataset = SMSegmentDataset(data=data, env_list=env_list, dataset_name="supervised", max_traj_length=max_traj_len, aux_provider_names=aux_provider_names)
        self.top_down_dataset = TopDownDataset(env_list=env_list,
                                               instr_negatives=False,
                                               instr_negatives_similar_only=False,
                                               seg_level=False,
                                               yaw_rand_range=0.0,
                                               img_w=img_w,
                                               img_h=img_h,
                                               map_w=map_w,
                                               map_h=map_h)
        self.seg_start_only = seg_start_only

    def __len__(self):
        return len(self.segment_dataset)

    def __getitem__(self, idx):
        data = self.segment_dataset[idx]

        # The data returned by the segment dataset includes instructions, actions, images and poses
        # We would like to add a top-down view of the scene for either every pose in the segment or only the first pose
        # Data is a dict, where each item is a list/batchtensor over timesteps of the entire env-long trajectory
        # For each of these timesteps, we would want to retrieve a top-down image to be used.
        # TODO: The data is always of size TRAJECTORY_LEN. Perhaps it's a good idea to have a variable batch size.

        top_down_images = []
        top_down_poses = []

        num_items = data["images"].size(0)
        if self.seg_start_only:
            prev_seg = {"env_id": -1, "set_idx": -1, "seg_idx": -1}
            for i in range(num_items):
                if data["md"][i] is None:
                    curr_seg = prev_seg
                else:
                    curr_seg = dict_slice(data["md"][i], ["env_id", "set_idx", "seg_idx"])
                if curr_seg != prev_seg:
                    prev_seg = curr_seg
                    top_down_image = self.top_down_dataset.get_top_down_image_env(curr_seg["env_id"])
                    #Presenter().show_image(top_down_image, "tdown_img", torch=True, waitkey=True)
                    top_down_images.append(top_down_image)
                else:
                    top_down_images.append(top_down_images[-1])
        else:
            ...
            #TODO: Implement the case where we get a rotated image for each timestep, not only the first one in the segment
            #EDIT: Nope, we're not doing that. TODO: Simplify this whole thing into a single function that loads an image of the env

        data["top_down_images"] = torch.cat(top_down_images, dim=0).unsqueeze(0)
        #data["top_down_poses"] = torch.cat(top_down_poses, dim=0).unsqueeze(0)
        return data

    def collate_fn(self, list_of_samples):

        # This will correctly collate everything, but the "env_image" that we added will become a list of tensors
        # instead of a single tensor. Turn it into a batch tensor
        data_t = self.segment_dataset.collate_fn(list_of_samples)
        data_t["top_down_images"] = Variable(torch.cat(data_t["top_down_images"], 0))
        #data_t["top_down_poses"] = Variable(torch.cat(data_t["top_down_poses"], 0))

        return data_t
