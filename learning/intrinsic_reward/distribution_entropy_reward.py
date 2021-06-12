import torch
from learning.intrinsic_reward.abstract_intrinsic_reward import AbstractIntrinsicReward


class DistributionEntropyReward(AbstractIntrinsicReward):
    def __init__(self, distribution_key, channel=None):
        super(DistributionEntropyReward, self).__init__()
        self.distribution_key = distribution_key
        self.channel = channel

    def get_reward(self, tensor_store):
        v_dist = tensor_store.get_latest_input(self.distribution_key)
        if self.channel:
            v_dist = v_dist[:, self.channel, :, :]

        entropy = -torch.sum(v_dist * torch.log(v_dist))
        return entropy