import numpy as np
import numpy.linalg as npl

from pomdp.reward.abstract_reward import AbstractReward

REWARD_COMPLETE = 1000

REWARD_STEP_BASELINE = 1    # Reward given for each trajectory step
REWARD_CLOSER = 0           # Reward proportional to distance towards next path point
REWARD_ITERATION = 0        # Reward given every iteration (-ve)
REWARD_TRUE_STOP = 1        # Reward given from stopping correctly at the end of segment
REWARD_SEGMENT = 0.5        # Reward given for correctly completing the segment
REWARD_FALSE_STOP = -1      # Reward given for stopping at the wrong time (-ve)
REWARD_OVERSHOOT = 0        # Reward given for not stopping after segment completed


NOMINAL_ADVANCEMENT = 0.1

SEG_END_THRESHOLD = 0.3
DIST_THRESHOLD = 0.2


class FollowPathReward(AbstractReward):

    def __init__(self, path):
        super(FollowPathReward, self).__init__(path)
        self.path = path
        self.curr_idx = 0
        self.last_pos = None
        self.complete = False
        self.end_idx = len(self.path) - 1

    def set_current_segment(self, start_idx, end_idx):
        if end_idx > len(self.path) - 1:
            end_idx = len(self.path) - 1
        self.end_idx = end_idx
        self.complete = False

    def get_reward(self, state, action):

        stopped = action[3] > 0.5

        reward = 0

        if self.curr_idx >= self.end_idx or self.curr_idx >= len(self.path):
            self.complete = True

        if stopped and not self.complete:
            self.complete = True
            reward += REWARD_FALSE_STOP

        if stopped and self.complete:
            reward += REWARD_TRUE_STOP

        if not stopped and self.complete:
            reward += REWARD_OVERSHOOT

        if self.complete:
            return reward, self.complete

        # Retrieve the locations of current and next points along the path
        curr_point_pos = self.path[self.curr_idx]
        next_point_pos = self.path[self.curr_idx + 1]
        seg_end_pos = self.path[self.end_idx - 1]

        curr_dir = next_point_pos - curr_point_pos
        curr_dir /= (npl.norm(curr_dir) + 1e-9)

        next_dot = np.dot(next_point_pos, curr_dir)
        curr_dot = np.dot(state.get_pos_2d(), curr_dir)

        curr_dist = npl.norm(state.get_pos_2d() - next_point_pos)

        #Give some reward for getting closer to the next point:
        if self.last_pos is not None:
            last_dist_seg_end = npl.norm(self.last_pos - seg_end_pos)
            curr_dist_seg_end = npl.norm(state.get_pos_2d() - seg_end_pos)
            advancement = last_dist_seg_end - curr_dist_seg_end
            reward_shaping = REWARD_CLOSER * advancement / NOMINAL_ADVANCEMENT
            reward += reward_shaping
        self.last_pos = state.get_pos_2d()


        #Then check if we have reached the next point and give step reward, higher if we are closer
        if (curr_dot > next_dot and curr_dist < DIST_THRESHOLD) or curr_dist < SEG_END_THRESHOLD:
            self.curr_idx += 1
            reward += REWARD_STEP_BASELINE * np.exp(-curr_dist ** 2)

        reward += REWARD_ITERATION

        if self.curr_idx == self.end_idx:
            self.complete = True
            reward += REWARD_SEGMENT
            #print (" segment complete!")

        #print ("get_reward: ", reward)
        return reward, self.complete
