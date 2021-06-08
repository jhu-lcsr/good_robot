import math
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import default_collate

class _DualDataLoaderIterator():
    def __init__(self, dual_dataloader):
        self.loader = dual_dataloader
        self.iter_a = iter(self.loader.loader_a)
        self.iter_b = iter(self.loader.loader_b)
        self.a_finished = False
        self.b_finished = False

    def get(self, it, finished):
        try:
            next_item = next(it)
        except StopIteration:
            # When loader A is finished, reset it
            it = iter(self.loader.loader_a)
            # If this throws exception too, that means the dataloder length is zero. In this case we don't want to catch it
            next_item = next(it)
            finished = True
        return next_item, finished, it

    def __next__(self):
        next_a, self.a_finished, self.iter_a = self.get(self.iter_a, self.a_finished)
        next_b, self.b_finished, self.iter_b = self.get(self.iter_b, self.b_finished)

        # Dataloader length is maximum of both datasets
        if self.loader.joint_length == "max":
            if self.a_finished and self.b_finished:
                raise StopIteration()
        # Dataloder length is minimum of both datasets
        elif self.loader.joint_length == "min":
            if self.a_finished or self.b_finished:
                raise StopIteration()
        # Dataloder length is infinte - reset the finished flags to False as soon as they become True
        elif self.loader.joint_length == "infinite":
            self.a_finished = False
            self.b_finished = False

        return next_a, next_b


class DualDataloader():

    def __init__(self, dataset_a, dataset_b, batch_size=1, shuffle=False,
                 sampler_a=None, sampler_b=None, batch_sampler_a=None, batch_sampler_b=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, joint_length="max"):
        """
        :param dataset_a:
        :param dataset_b:
        :param batch_size:
        :param shuffle:
        :param sampler_a:
        :param sampler_b:
        :param batch_sampler_a:
        :param batch_sampler_b:
        :param num_workers:
        :param collate_fn:
        :param pin_memory:
        :param drop_last:
        :param timeout:
        :param worker_init_fn:
        :param joint_length: either "max", "min" or "infinite"
        """

        if hasattr(dataset_a, "collate_fn"):
            collate_a = dataset_a.collate_fn
        else:
            collate_a = collate_fn
        if hasattr(dataset_b, "collate_fn"):
            collate_b = dataset_a.collate_fn
        else:
            collate_b = collate_fn

        self.loader_a = DataLoader(dataset_a,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   sampler=sampler_a,
                                   batch_sampler=batch_sampler_a,
                                   num_workers=num_workers,
                                   collate_fn=collate_a,
                                   pin_memory=pin_memory,
                                   drop_last=drop_last,
                                   timeout=timeout,
                                   worker_init_fn=worker_init_fn)

        self.loader_b = DataLoader(dataset_b,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   sampler=sampler_b,
                                   batch_sampler=batch_sampler_b,
                                   num_workers=num_workers,
                                   collate_fn=collate_b,
                                   pin_memory=pin_memory,
                                   drop_last=drop_last,
                                   timeout=timeout,
                                   worker_init_fn=worker_init_fn)

        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.length_a = len(self.loader_a)
        self.length_b = len(self.loader_b)
        self.joint_length = joint_length

    def __len__(self):
        if self.joint_length == "max":
            return max(self.length_a, self.length_b)
        elif self.joint_length == "min":
            return min(self.length_a, self.length_b)
        elif self.joint_length == "infinity":
            return math.inf

    def __iter__(self):
        return _DualDataLoaderIterator(self)
