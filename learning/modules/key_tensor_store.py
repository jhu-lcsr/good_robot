import torch
from torch.autograd import Variable
from learning.inputs.partial_2d_distribution import Partial2DDistribution
from learning.utils import save_tensor_as_img_during_rollout, get_viz_dir_for_rollout

def save_tensors_as_images(tensor_store, list_of_keys, prefix):
    for key in list_of_keys:
        tensor = tensor_store.get_inputs_batch(key, cat_not_stack=True)
        if tensor is None:
            print(f"save_tensors_as_images: Tensor {key} not present in tensor store")
            continue
        if len(tensor.shape) not in [3,4]:
            print(f"save_tensors_as_images: Tensor {key} has unsupported shape: {tensor.shape}")
            continue

        # If it's a batch of images/feature maps, take the last one
        if len(tensor.shape) == 4:
            tensor = tensor[-1]
        # If the remaining image/feature map has more than 3 channels, take the first 3
        if len(tensor.shape) == 3 and tensor.shape[0] > 3:
            tensor = tensor[0:3, :, :]

        save_tensor_as_img_during_rollout(tensor, key, prefix)


class KeyTensorStore():
    def __init__(self):
        self.tensors = {}
        self.flags = {}

    def reset(self):
        self.tensors = {}
        self.flags = {}

    def cuda(self, device=None):
        for k, v in self.tensors.items():
            v.cuda(device)

    def set_flag(self, key, value):
        self.flags[key] = value

    def get_flag(self, key):
        return self.flags.get(key)

    def to(self, device=None):
        for k, v in self.tensors.items():
            v.cuda(device)

    def append(self, other):
        for k,tlist in other.tensors.items():
            assert isinstance(tlist, list)
            if k not in self.tensors:
                self.tensors[k] = []
            self.tensors[k] += tlist

    def keep_input(self, key, input):
        """
        Stores a tensor for later retrieval with a given key
        :param key:
        :param input:
        :return:
        """
        if key not in self.tensors  :
            self.tensors[key] = []
        self.tensors[key].append(input)

    def keep_inputs(self, key, input):
        """
        Stores a batch or sequence of tensors for later retrieval with a given key
        :param key:
        :param input:
        :return:
        """
        if type(input) == Variable or type(input) == torch.Tensor:
            for i in range(input.size(0)):
                self.keep_input(key, input[i:i+1])
        elif isinstance(input, Partial2DDistribution):
            self.keep_input(key, input)
        elif type(input) == list:
            for i in range(len(input)):
                inp = input[i]
                if type(inp) is Variable:
                    inp = inp.unsqueeze(0)
                self.keep_input(key, inp)
        else:
            raise Exception("ModuleWithAuxiliaries: Unrecognized input: " + str(type(input)))

    def get(self, key):
        if key in self.tensors:
            return self.tensors[key]
        return None

    def get_latest_input(self, key):
        """
        Retrieves a the latest previously stored tensor with the given key
        :param key:
        :return:
        """
        if key in self.tensors:
            return self.tensors[key][-1]
        return None

    def get_inputs_batch(self, key, cat_not_stack=False):
        """
        Retrieves all tensors with the given key, stacked in batch
        :param key:
        :return:
        """
        if key not in self.tensors:
            return None

        v = self.tensors[key]
        if isinstance(v[0], Partial2DDistribution):
            return v

        if cat_not_stack:
            return torch.cat(self.tensors[key], dim=0)
        else:
            return torch.stack(self.tensors[key], dim=0)

    def clear_inputs(self, key):
        """
        Removes all stored tensors associated with the given key
        :param key:
        :return:
        """
        if key in self.tensors:
            del self.tensors[key]
