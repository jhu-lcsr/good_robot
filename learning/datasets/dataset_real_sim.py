import cv2
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


class RealSimDataset(Dataset):
    def __init__(self, real_dataset, sim_dataset, aligned=False):
        self.real_dataset = real_dataset
        self.sim_dataset = sim_dataset
        self.aligned = aligned
        if aligned:
            assert(len(self.sim_dataset) == len(self.real_dataset))

    def __len__(self):
        return min(len(self.real_dataset), len(self.sim_dataset))

    def __getitem__(self, index):
        if self.aligned:
            seed = np.random.randint(0, 2**32)
            np.random.seed(seed)
        sample_real = self.real_dataset[index]
        if self.aligned:
            np.random.seed(seed)
        sample_sim = self.sim_dataset[index]
        out_dict = {"real": sample_real, "sim": sample_sim}
        return out_dict

    def collate_fn(self, list_of_samples):
        out_tuple_real = self.real_dataset.collate_fn([sample["real"] for sample in list_of_samples])
        out_tuple_sim = self.real_dataset.collate_fn([sample["sim"] for sample in list_of_samples])
        out_dict = {"real": out_tuple_real, "sim": out_tuple_sim}
        # 1 datasample is 1 real sample + 1 simulated sample. This enables to have the same number of each.
        return out_dict


class ConcatRealSimDataset(Dataset):
    def __init__(self, real_dataset, sim_dataset):
        self.real_dataset = real_dataset
        self.sim_dataset = sim_dataset
        self.nreal = len(real_dataset)
        self.nsim = len(sim_dataset)

    def __len__(self):
        return len(self.real_dataset)+len(self.sim_dataset)

    def __getitem__(self, index):
        if index < self.nreal:
            out = self.real_dataset[index]
            out['label'] = 1.
        else:
            out = self.sim_dataset[index-self.nreal]
            out['label'] = 0.
        return out

    def collate_fn(self, list_of_samples):
        labels = [sample['label'] for sample in list_of_samples]
        out_tuple = self.real_dataset.collate_fn(list_of_samples)
        out_tuple += (labels,)
        return out_tuple