import numpy as np
from pomdp.reward.abstract_reward import AbstractReward

CORRECT_STOP_DISTANCE = 1.0


class StopCorrectlyReward(AbstractReward):

    def __init__(self, path):
        super(StopCorrectlyReward, self).__init__(path)
        self.path = path

    def get_reward(self, state, action, done_now):
        if len(self.path) == 0:
            return 0
        dst_to_endpt = np.linalg.norm(self.path[-1] - state.get_pos_2d())
        correct_stop = dst_to_endpt < CORRECT_STOP_DISTANCE

        stop_reward = max((CORRECT_STOP_DISTANCE - dst_to_endpt) / CORRECT_STOP_DISTANCE, -0.5)

        # Return reward 1 if stopped in correct goal region.
        # Reward -0.5 if stopped outside of correct goal region
        status = ""
        if action[3] > 0.5:
            status = "PASS" if correct_stop else "FAIL"
            #reward = 1.0 if correct_stop else -0.5
            reward = stop_reward
        # Return negative reward if didn't stop and flew off
        elif done_now:
            reward = -0.5
        # Otherwise return 0
        else:
            reward = 0.0
        #print(f"StopCorrectlyReward: {reward} {status}")
        return reward
