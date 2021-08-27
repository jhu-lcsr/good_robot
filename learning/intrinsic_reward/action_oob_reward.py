import torch
from learning.intrinsic_reward.abstract_intrinsic_reward import AbstractIntrinsicReward

import transformations

class ActionOutOfBoundsReward(AbstractIntrinsicReward):
    def __init__(self):
        super(ActionOutOfBoundsReward, self).__init__()
        self.min_vel_x = 0.2
        self.max_vel_x = 1.0
        self.min_yaw_rate = -1.3
        self.max_yaw_rate = 1.3
        self.penalty_strength = 1.0
        self.oob_allowance = 0.7

    def reset(self):
        pass

    def get_reward(self, action):
        x_vel = action[0]
        yawrate = action[2]

        x_vel_upper_margin = max(x_vel - (self.max_vel_x + self.oob_allowance), 0)
        x_vel_lower_margin = max((self.min_vel_x - self.oob_allowance) - x_vel, 0)
        yawrate_upper_margin = max(yawrate - (self.max_yaw_rate + self.oob_allowance), 0)
        yawrate_lower_margin = max((self.min_yaw_rate - self.oob_allowance) - yawrate, 0)

        x_vel_margin = max(x_vel_lower_margin, x_vel_upper_margin)
        yawrate_margin = max(yawrate_lower_margin, yawrate_upper_margin)

        penalty = (x_vel_margin + yawrate_margin) * self.penalty_strength
        penalty = min(penalty, 2.0)
        reward = -penalty
        return reward