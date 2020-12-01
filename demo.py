import numpy as np
import cv2
import os
from utils import ACTION_TO_ID

class Demonstration():
    def __init__(self, path, demo_num, check_z_height, task_type='stack'):
        # path is expected to be <logs/exp_name>
        self.action_log = np.loadtxt(os.path.join(path, 'transitions',
            'executed-actions-0.log.txt'))
        self.rgb_dir = os.path.join(path, 'data', 'color-heightmaps')
        self.depth_dir = os.path.join(path, 'data', 'depth-heightmaps')
        self.image_names = sorted(os.listdir(self.rgb_dir))
        self.demo_num = demo_num
        self.check_z_height = check_z_height
        self.task_type = task_type

        if self.task_type == 'stack':
            # populate actions in dict keyed by stack height {stack_height : {action : (x, y, z, theta)}}
            self.action_dict = {}
            for s in range(3):
                # store push, grasp, and place actions for demo at stack height s
                # TODO(adit98) figure out how to incorporate push actions into this paradigm
                # TODO(adit98) note this assumes perfect demo
                # if stack height is 0, indices 0 and 1 of the action log correspond to grasp and place respectively
                demo_first_ind = 2 * s
                self.action_dict[s] = {ACTION_TO_ID['grasp'] : self.action_log[demo_first_ind],
                        ACTION_TO_ID['place'] : self.action_log[demo_first_ind + 1]}

        elif self.task_type == 'unstack':
            # get number of actions in demo
            self.num_actions = len(self.action_log)

            # populate actions in dict keyed by stack height {stack_height : {action : (x, y, z, theta)}}
            self.action_dict = {}
            for s in range(1, 5):
                # store push, grasp, and place actions for demo at stack height s
                demo_ind = -2 * (5 - s)
                self.action_dict[s] = {ACTION_TO_ID['grasp'] : self.action_log[demo_ind],
                        ACTION_TO_ID['place'] : self.action_log[demo_ind + 1],
                        'demo_ind': self.num_actions + demo_ind}

        else:
            # task type is some grasp-place sequence
            # get number of actions in demo
            self.num_actions = len(self.action_log)

            # populate actions in dict keyed by action_pair number {action_pair : {action : (x, y, z, theta)}}
            # divide num actions by 2 to get number of grasp/place pairs
            self.action_dict = {}
            for action_pair in range(self.num_actions // 2):
                demo_ind = action_pair * 2
                grasp_image_ind = int(self.image_names[demo_ind].split('.')[0])
                place_image_ind = int(self.image_names[demo_ind + 1].split('.')[0])
                self.action_dict[action_pair] = {ACTION_TO_ID['grasp'] : self.action_log[demo_ind],
                        ACTION_TO_ID['place'] : self.action_log[demo_ind + 1],
                        'grasp_image_ind': grasp_image_ind, 'place_image_ind': place_image_ind}

    def get_heightmaps(self, action_str, stack_height):
        # e.g. initial rgb filename is 000000.orig.color.png, only for stack demos
        if action_str != 'orig' and self.task_type == 'stack':
            action_str = str(stack_height) + action_str

        rgb_filename = os.path.join(self.rgb_dir,
                '%06d.%s.color.png' % (stack_height, action_str))
        depth_filename = os.path.join(self.depth_dir,
                '%06d.%s.depth.png' % (stack_height, action_str))
        print("Processing:", rgb_filename, depth_filename)

        rgb_heightmap = cv2.cvtColor(cv2.imread(rgb_filename), cv2.COLOR_BGR2RGB)
        depth_heightmap = cv2.imread(depth_filename, -1).astype(np.float32)/100000

        return rgb_heightmap, depth_heightmap

    def get_action(self, workspace_limits, primitive_action, stack_height, stack_trainer=None,
            row_trainer=None):
        # ensure one of stack trainer or row trainer is provided
        if stack_trainer is None and row_trainer is None:
            raise ValueError("Must provide one of stack_trainer or row_trainer")

        # TODO(adit98) clean up the way demo heightmaps are saved to reduce confusion
        # set action_str based on primitive action
        # heightmap_height is the height we use to get the demo heightmaps
        if self.task_type == 'stack':
            if not self.check_z_height:
                # if we completed a stack, prev_stack_height will be 4, but we want the imitation actions for stack height 1
                stack_height = (stack_height - 1) if stack_height < 4 else 0
            else:
                stack_height = np.round(stack_height).astype(int)
                stack_height = (stack_height - 1) if stack_height < 4 else 0

            # TODO(adit98) deal with push
            if primitive_action == 'push':
                return -1

            if stack_height == 0 and primitive_action == 'grasp':
                action_str = 'orig'
            elif primitive_action == 'grasp':
                # if primitive action is grasp, we need the previous place heightmap and grasp action
                action_str = 'place'
                heightmap_height -= 1
            else:
                # if primitive action is place, get the previous grasp heightmap
                action_str = 'grasp'

            color_heightmap, valid_depth_heightmap = self.get_heightmaps(action_str, stack_height)
            action_str = primitive_action

        elif self.task_type == 'unstack':
            if primitive_action == 'grasp':
                # offset is 2 for stack height 4, 4 for stack height 3, ...
                #offset = 10 - 2 * stack_height
                color_heightmap, valid_depth_heightmap = self.get_heightmaps(primitive_action, self.action_dict[stack_height]['demo_ind'])

            elif primitive_action == 'place':
                # offset is grasp_offset - 1 because place is always 1 action after grasp
                #offset = 9 - 2 * stack_height
                color_heightmap, valid_depth_heightmap = self.get_heightmaps(primitive_action, self.action_dict[stack_height]['demo_ind'] + 1)
        else:
            if primitive_action == 'grasp':
                color_heightmap, valid_depth_heightmap = self.get_heightmaps(primitive_action,
                        self.action_dict[stack_height]['grasp_image_ind'])

            elif primitive_action == 'place':
                color_heightmap, valid_depth_heightmap = self.get_heightmaps(primitive_action,
                        self.action_dict[stack_height]['place_image_ind'])

        # get stack features if stack_trainer is provided
        if stack_trainer is not None:
            # to get vector of 64 vals, run trainer.forward with get_action_feat
            stack_push, stack_grasp, stack_place = stack_trainer.forward(color_heightmap,
                    valid_depth_heightmap, is_volatile=True, keep_action_feat=True, use_demo=True)

        # get row features if row_trainer is provided
        if row_trainer is not None:
            # to get vector of 64 vals, run trainer.forward with get_action_feat
            row_push, row_grasp, row_place = row_trainer.forward(color_heightmap,
                    valid_depth_heightmap, is_volatile=True, keep_action_feat=True, use_demo=True)

        # get demo action index vector
        action_vec = self.action_dict[stack_height][ACTION_TO_ID[primitive_action]]

        # convert rotation angle to index
        best_rot_ind = np.around((np.rad2deg(action_vec[-2]) % 360) * 16 / 360).astype(int)

        # convert robot coordinates to pixel
        workspace_pixel_offset = workspace_limits[:2, 0] * -1 * 1000
        best_action_xy = ((workspace_pixel_offset + 1000 * action_vec[:2]) / 2).astype(int)

        # need to swap x and y coordinates for best_action_xy
        best_action_xy = [best_action_xy[1], best_action_xy[0]]

        # initialize best actions for stacking and row making
        best_action_stack, best_action_row = None, None

        # index predictions to obtain best action
        if primitive_action == 'grasp':
            if stack_trainer is not None:
                best_action_stack = stack_grasp[best_rot_ind, :, best_action_xy[0],
                        best_action_xy[1]]
            
            if row_trainer is not None:
                best_action_row = row_grasp[best_rot_ind, :, best_action_xy[0],
                        best_action_xy[1]]

        elif primitive_action == 'place':
            if stack_trainer is not None:
                best_action_stack = stack_place[best_rot_ind, :, best_action_xy[0],
                        best_action_xy[1]]
            
            if row_trainer is not None:
                best_action_row = row_place[best_rot_ind, :, best_action_xy[0],
                        best_action_xy[1]]

        # return best action for row and stack (None if only using 1 or the other) and action
        return best_action_row, best_action_stack, ACTION_TO_ID[primitive_action]