import torch
from learning.intrinsic_reward.abstract_intrinsic_reward import AbstractIntrinsicReward


class MapCoverageReward(AbstractIntrinsicReward):
    def __init__(self):
        super(MapCoverageReward, self).__init__()
        #self.map_key = map_key
        #self.threshold = threshold
        self.prev_potential = None

    def reset(self):
        self.prev_potential = None

    def get_reward(self, coverage_w):
        #map = tensor_store.get_latest_input(self.map_key)
        #ones_mask = (map != -1000).long()
        #coverage_mask = (map.abs() > self.threshold).long()
        ones_mask = torch.ones_like(coverage_w)
        frac_coverage = coverage_w.sum() / (ones_mask.sum() + 1e-20)

        if self.prev_potential is None:
            self.prev_potential = frac_coverage

        reward = (frac_coverage - self.prev_potential).detach().item()
        self.prev_potential = frac_coverage
        return reward