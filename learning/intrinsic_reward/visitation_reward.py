import torch
from learning.intrinsic_reward.abstract_intrinsic_reward import AbstractIntrinsicReward

import transformations

MIN_START_STOP_DIST_PX = 5.0

class VisitationReward(AbstractIntrinsicReward):
    def __init__(self, world_size_m, world_size_px):
        super(VisitationReward, self).__init__()
        self.world_size_m = world_size_m
        self.world_size_px = world_size_px
        self.prev_potential = None
        self.start_best_stop_dist = None

        self.visit_alpha = 1.0
        self.stop_alpha = 2.0
        self.stop_offset = 0.0

    def reset(self):
        self.prev_potential = None
        self.start_best_stop_dist = None

    def get_reward(self, v_dist_w, cam_pos, action):
        # If stopped:
        pos_in_map_m = cam_pos[0:1, 0:2]# * self.world_size_px / self.

        pos_in_map_px = torch.from_numpy(transformations.pos_m_to_px(pos_in_map_m.detach().cpu().numpy(),
                                                     self.world_size_px,
                                                     self.world_size_m,
                                                     self.world_size_px))

        pos_x = int(pos_in_map_px[0, 0].item() + 0.5)
        pos_y = int(pos_in_map_px[0, 1].item() + 0.5)

        visit_dist = v_dist_w[0, 0, :, :]
        stop_dist = v_dist_w[0, 1, :, :]

        #TODO: Consider this. This way the total reward that can be collected is 1
        visit_dist -= visit_dist.min()
        visit_dist /= (visit_dist.max() + 1e-10)
        stop_dist -= stop_dist.min()
        stop_dist /= (stop_dist.max() + 1e-10)

        pos_x = min(max(pos_x, 0), visit_dist.shape[0] - 1)
        pos_y = min(max(pos_y, 0), visit_dist.shape[1] - 1)

        visit_prob = visit_dist[pos_x, pos_y].item()
        stop_prob = stop_dist[pos_x, pos_y].item()

        # No batch dimension here:
        max_stop_prob, argmax_stop_prob = stop_dist.view(-1).max(0)
        best_stop_pos_x = int(argmax_stop_prob / stop_dist.shape[0])
        best_stop_pos_y = int(argmax_stop_prob % stop_dist.shape[0])

        best_stop_pos = torch.Tensor([best_stop_pos_x, best_stop_pos_y])
        pos = torch.Tensor([pos_x, pos_y])
        dst_to_best_stop = torch.norm(pos - best_stop_pos)

        if self.start_best_stop_dist is None:
            self.start_best_stop_dist = min(dst_to_best_stop, MIN_START_STOP_DIST_PX)

        visit_potential = self.visit_alpha * visit_prob
        # THIS IS NOT POTENTIAL NOW
        # TODO: Change terminology
        if self.prev_potential is None:
            self.prev_potential = visit_potential
            # Don't give reward for the first step
            visit_reward = visit_potential * 0
        # Give reward for visiting the high-probability states at next timestep
        else:
            visit_reward = visit_potential - self.prev_potential
            self.prev_potential = visit_potential

        if action[3] > 0.5:
            stop_reward_a = (stop_prob - self.stop_offset) * self.stop_alpha
            stop_reward_b = 0.2 - min(dst_to_best_stop / (self.start_best_stop_dist + 1e-9), 1)
            stop_reward = stop_reward_a + stop_reward_b
        else:
            stop_reward = 0.0

        #total_reward = visit_reward + stop_reward
        return visit_reward, stop_reward