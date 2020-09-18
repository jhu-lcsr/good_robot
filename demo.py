import numpy as np
import cv2
import os
from utils import ACTION_TO_ID

class Demonstration():
    def __init__(self, path, demo_num):
        # path is expected to be <logs/exp_name>
        self.action_log = np.loadtxt(os.path.join(path, 'transitions',
            'executed-actions-0.log.txt'))
        self.rgb_dir = os.path.join(path, 'data', 'color-heightmaps')
        self.depth_dir = os.path.join(path, 'data', 'depth-heightmaps')
        self.demo_num = demo_num

        # this str is for loading the correct images, it will be adjusted based on selected action
        self.action_str = 'orig'

        # populate actions in dict keyed by stack height {stack_height : {action : (x, y, z, theta)}}
        self.action_dict = {}
        for s in range(1, 4):
            # store push, grasp, and place actions for demo at stack height s
            # TODO(adit98) figure out how to incorporate push actions into this paradigm
            # TODO(adit98) note this assumes perfect demo
            # if stack height is 1, indices 0 and 1 of the action log correspond to grasp and place respectively
            demo_first_ind = 2 * (s - 1)
            self.action_dict[s] = {ACTION_TO_ID['grasp'] : self.action_log[demo_first_ind],
                    ACTION_TO_ID['place'] : self.action_log[demo_first_ind + 1]}

    def get_heightmaps(self, action_str, stack_height):
        # e.g. initial rgb filename is 000000.orig.color.png
        if action_str != 'orig':
            action_str = str(int(stack_height) - 1) + action_str

        rgb_filename = os.path.join(self.rgb_dir, 
                '%06d.%s.color.png' % (self.demo_num, action_str))
        depth_filename = os.path.join(self.depth_dir,
                '%06d.%s.depth.png' % (self.demo_num, action_str))

        rgb_heightmap = cv2.cvtColor(cv2.imread(rgb_filename), cv2.COLOR_BGR2RGB)
        depth_heightmap = cv2.imread(depth_filename, -1).astype(np.float32)/100000

        return rgb_heightmap, depth_heightmap

    # TODO(adit98) figure out how to get primitive action
    # TODO(adit98) this will NOT work for novel tasks, worry about that later
    def get_action(self, trainer, workspace_limits, primitive_action, stack_height):
        # if we completed a stack, prev_stack_height will be 4, but we want the imitation actions for stack height 1
        # TODO(adit98) switched this to get nonlocal_variables['stack_height'] now, so see how it is different
        stack_height = stack_height if stack_height < 4 else 1

        # TODO(adit98) deal with push
        if primitive_action == 'push':
            return -1

        # set action_str based on primitive action
        if stack_height == 1 and primitive_action == 'grasp':
            action_str = 'orig'
        elif primitive_action == 'grasp':
            action_str = 'grasp'
        else:
            action_str = 'place'

        color_heightmap, valid_depth_heightmap = self.get_heightmaps(action_str, stack_height)
        # to get vector of 64 vals, run trainer.forward with get_action_feat
        push_preds, grasp_preds, place_preds = trainer.forward(color_heightmap,
                valid_depth_heightmap, is_volatile=True, keep_action_feat=True, use_demo=True)
        action_vec = self.action_dict[stack_height][ACTION_TO_ID[primitive_action]]

        # TODO(adit98) figure out how to convert rotation angle to index
        best_rot_ind = np.around(np.rad2deg(action_vec[-2]) * 16 / 360).astype(int)

        # TODO(adit98) convert robot coordinates to pixel
        workspace_pixel_offset = workspace_limits[:2, 0] * -1 * 1000
        best_action_xy = ((workspace_pixel_offset + 1000 * action_vec[:2]) / 2).astype(int)

        # TODO(adit98) figure out if we want more than 1 coordinate
        # TODO(adit98) add logic for pushing here
        if primitive_action == 'grasp':
            best_action = grasp_preds[best_rot_ind, :, best_action_xy[0], best_action_xy[1]]

        # TODO(adit98) find out why place preds inds were different before
        elif primitive_action == 'place':
            best_action = place_preds[best_rot_ind, :, best_action_xy[0], best_action_xy[1]]

        return best_action, ACTION_TO_ID[primitive_action]
