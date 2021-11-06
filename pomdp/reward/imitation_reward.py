import numpy as np
from policies.fancy_carrot_planner import FancyCarrotPlanner
from policies.simple_carrot_planner import SimpleCarrotPlanner
from pomdp.reward.abstract_reward import AbstractReward

from parameters.parameter_server import get_current_parameters

class ImitationReward(AbstractReward):

    def __init__(self, path):
        super(ImitationReward, self).__init__(path)
        self.path = path
        self.curr_idx = 0
        self.last_pos = None
        #currentparams = get_current_parameters()
        #if currentparams is not None:
        #    self.params = get_current_parameters()["RolloutParams"]
        #    self.ref_policy = FancyCarrotPlanner(path) if self.params["OracleType"] == "FancyCarrotPlanner" else SimpleCarrotPlanner(path)
        #else:
        #    self.ref_policy = None
        self.end_idx = len(self.path) - 1

    def set_current_segment(self, start_idx, end_idx):
        pass
        #self.ref_policy.set_current_segment(start_idx, end_idx)

    def get_reward(self, state, action):
        return 0
        if self.ref_policy is None:
            return 0
        ref_action = self.ref_policy.get_action(state)
        if ref_action is None or action is None:
            return 0
        reward = -np.linalg.norm(ref_action[0:3] - action[0:3])
        reward = 0 #broken
        return reward
