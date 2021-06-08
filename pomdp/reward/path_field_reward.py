import numpy as np
from pomdp.reward.abstract_reward import AbstractReward


class FollowPathFieldReward(AbstractReward):

    def __init__(self, path):
        super(FollowPathFieldReward, self).__init__(path)
        self.path = path
        self.end_idx = len(self.path) - 1
        self.prev_potential = None

    def _calc_potential(self, curr_pos):
        if len(self.path) == 0:
            return 0
        distances = np.asarray([np.linalg.norm(curr_pos - p) for p in self.path])
        closest_pt_idx = np.argmin(distances)
        closest_pt_dst = distances[closest_pt_idx]

        path_term = closest_pt_dst
        goal_term = closest_pt_idx / len(self.path)

        potential = path_term + goal_term
        return potential

    def get_reward(self, state, action, done_now):
        curr_pos = state.get_pos_2d()
        potential = self._calc_potential(curr_pos)

        if self.prev_potential is None:
            self.prev_potential = potential

        reward = potential - self.prev_potential
        self.prev_potential = potential

        #print(f"FollowPathFieldReward: {reward}")

        return reward
