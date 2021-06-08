import torch
from learning.intrinsic_reward.abstract_intrinsic_reward import AbstractIntrinsicReward


class CorrectGoalPredictionReward(AbstractIntrinsicReward):
    def __init__(self, distribution_key, channel=None):
        super(CorrectGoalPredictionReward, self).__init__()
        self.distribution_key = distribution_key
        self.channel = channel

    def get_reward(self, tensor_store):
        v_dist = tensor_store.get_latest_input(self.distribution_key)
        if self.channel:
            goal_dist = v_dist[:, self.channel, :, :]
        else:
            goal_dist = v_dist

        # TODO: Complete this
        return entropy