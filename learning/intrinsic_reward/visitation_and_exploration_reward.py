import torch
from learning.intrinsic_reward.abstract_intrinsic_reward import AbstractIntrinsicReward

import transformations

MIN_START_STOP_DIST_PX = 5.0

class VisitationAndExplorationReward(AbstractIntrinsicReward):
    def __init__(self, world_size_m, world_size_px):
        super(VisitationAndExplorationReward, self).__init__()
        self.world_size_m = world_size_m
        self.world_size_px = world_size_px
        self.prev_potential = None
        self.start_best_stop_dist = None
        self.prev_goal_visible_prob = None

        self.visit_alpha = 1.0
        self.stop_alpha = 3.0
        self.exploration_alpha = 3.0
        self.stop_offset = -0.1

    def reset(self):
        self.prev_potential = None
        self.start_best_stop_dist = None
        self.prev_goal_visible_prob = None

    def get_reward(self, v_dist_w, goal_oob_prob_w, cam_pos, action):
        # Prepare things
        pos_in_map_m = cam_pos[0:1, 0:2]# * self.world_size_px / self.
        pos_in_map_px = torch.from_numpy(transformations.pos_m_to_px(pos_in_map_m.detach().cpu().numpy(),
                                                     self.world_size_px,
                                                     self.world_size_m,
                                                     self.world_size_px))

        pos_x = int(pos_in_map_px[0, 0].item() + 0.5)
        pos_y = int(pos_in_map_px[0, 1].item() + 0.5)

        visit_dist = v_dist_w[0, 0, :, :]
        partial_stop_dist = v_dist_w[0, 1, :, :]
        outside_stop_prob = goal_oob_prob_w.item()
        goal_visible_prob = 1 - outside_stop_prob

        pos_x = min(max(pos_x, 0), visit_dist.shape[0] - 1)
        pos_y = min(max(pos_y, 0), visit_dist.shape[1] - 1)

        # -----------------------------------------------------------------------
        # Calculate visitation reward (potential shaped by visitation probability)

        #TODO: Consider this. This way the total reward that can be collected is 1
        visit_dist -= visit_dist.min()
        visit_dist /= (visit_dist.max() + 1e-10)
        visit_prob = visit_dist[pos_x, pos_y].item()

        # Give reward for visiting the high-probability states at next timestep
        visit_potential = self.visit_alpha * visit_prob
        if self.prev_potential is None:
            self.prev_potential = visit_potential
        visit_reward = visit_potential - self.prev_potential
        self.prev_potential = visit_potential

        # -----------------------------------------------------------------------
        # Calculate stop reward consisting of 2 terms:
        #  Term A: Reward proportional to the goal probability
        #  Term B: Reward proportional to the negative distance to most likely goal location, weighed by the probability that t

        # TODO: Consider this re-normalization approach and if it's any good
        partial_stop_dist -= partial_stop_dist.min()
        partial_stop_dist /= (partial_stop_dist.max() + 0.01)
        #partial_stop_dist *= goal_visible_prob

        # No batch dimension here:
        stop_prob_at_pos = partial_stop_dist[pos_x, pos_y].item()
        max_stop_prob, argmax_stop_prob = partial_stop_dist.view(-1).max(0)
        best_stop_pos_x = int(argmax_stop_prob / partial_stop_dist.shape[0])
        best_stop_pos_y = int(argmax_stop_prob % partial_stop_dist.shape[0])

        best_stop_pos = torch.Tensor([best_stop_pos_x, best_stop_pos_y])
        pos = torch.Tensor([pos_x, pos_y])
        dst_to_best_stop = torch.norm(pos - best_stop_pos)

        if self.start_best_stop_dist is None:
            self.start_best_stop_dist = min(dst_to_best_stop, MIN_START_STOP_DIST_PX)

        if action[3] > 0.5:
            # Term A
            stop_reward_a = (stop_prob_at_pos - self.stop_offset) * self.stop_alpha

            # Term B
            stop_reward_b_raw = 0.2 - min(dst_to_best_stop / (self.start_best_stop_dist + 1e-9), 1)
            #stop_reward_b = stop_reward_b_raw * goal_visible_prob
            stop_reward_b = stop_reward_b_raw
            stop_reward = stop_reward_a + stop_reward_b
        else:
            stop_reward = 0.0

        # -----------------------------------------------------------------------
        # Calculate exploration reward, using probability that goal is observed as a potential function

        if self.prev_goal_visible_prob is None:
            self.prev_goal_visible_prob = goal_visible_prob
        exploration_reward = (goal_visible_prob - self.prev_goal_visible_prob) * self.exploration_alpha
        self.prev_goal_visible_prob = goal_visible_prob

        # -----------------------------------------------------------------------
        return visit_reward, stop_reward, exploration_reward