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

            # start at 1 since the structure starts with size 1
            for action_pair in range(1, (self.num_actions // 2) + 1):
                demo_ind = (action_pair - 1) * 2
                grasp_image_ind = int(self.image_names[demo_ind].split('.')[0])
                place_image_ind = int(self.image_names[demo_ind + 1].split('.')[0])
                self.action_dict[action_pair] = {ACTION_TO_ID['grasp'] : self.action_log[demo_ind],
                        ACTION_TO_ID['place'] : self.action_log[demo_ind + 1],
                        'grasp_image_ind': grasp_image_ind, 'place_image_ind': place_image_ind}

    def get_heightmaps(self, action_str, stack_height, use_hist=False, history_len=3):
        # e.g. initial rgb filename is 000000.orig.color.png, only for stack demos
        if action_str != 'orig' and self.task_type == 'stack':
            action_str = str(stack_height) + action_str

        rgb_filename = os.path.join(self.rgb_dir,
                '%06d.%s.color.png' % (stack_height, action_str))
        depth_filename = os.path.join(self.depth_dir,
                '%06d.%s.depth.png' % (stack_height, action_str))

        # read rgb and depth heightmap
        rgb_heightmap = cv2.cvtColor(cv2.imread(rgb_filename), cv2.COLOR_BGR2RGB)
        depth_heightmap = cv2.imread(depth_filename, -1).astype(np.float32)/100000

        # if using history, need to modify depth heightmap
        if use_hist:
            depth_heightmap_history = [depth_heightmap]
            image_ind = self.image_names.index(rgb_filename.split('/')[-1])
            hist_ind = image_ind

            # iterate through last history_len frames and add to list
            for i in range(history_len - 1):
                # calculate previous index
                hist_ind = max(0, hist_ind - 1)

                # load heightmap and add to list
                heightmap_path = os.path.join(self.depth_dir, self.image_names[image_ind].replace('color', 'depth'))
                hist_depth = cv2.imread(heightmap_path, -1).astype(np.float32)/100000
                depth_heightmap_history.append(hist_depth)

            return rgb_heightmap, np.stack(depth_heightmap_history, axis=-1)

        return rgb_heightmap, np.stack([depth_heightmap] * 3, axis=-1)

    def get_action(self, workspace_limits, primitive_action, stack_height, stack_trainer=None,
            row_trainer=None, unstack_trainer=False, vertical_square_trainer=False, use_hist=False):
        # ensure one of stack trainer or row trainer is provided
        if stack_trainer is None and row_trainer is None and unstack_trainer is None and vertical_square_trainer is None:
            raise ValueError("Must provide at least one trainer")

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

            color_heightmap, valid_depth_heightmap = self.get_heightmaps(action_str,
                    stack_height, use_hist=use_hist)
            action_str = primitive_action

        elif self.task_type == 'unstack':
            if primitive_action == 'grasp':
                # offset is 2 for stack height 4, 4 for stack height 3, ...
                #offset = 10 - 2 * stack_height
                color_heightmap, valid_depth_heightmap = self.get_heightmaps(primitive_action,
                        self.action_dict[stack_height]['demo_ind'], use_hist=use_hist)

            elif primitive_action == 'place':
                # offset is grasp_offset - 1 because place is always 1 action after grasp
                #offset = 9 - 2 * stack_height
                color_heightmap, valid_depth_heightmap = self.get_heightmaps(primitive_action,
                        self.action_dict[stack_height]['demo_ind'] + 1, use_hist=use_hist)
        else:
            if primitive_action == 'grasp':
                color_heightmap, valid_depth_heightmap = self.get_heightmaps(primitive_action,
                        self.action_dict[stack_height]['grasp_image_ind'], use_hist=use_hist)

            elif primitive_action == 'place':
                color_heightmap, valid_depth_heightmap = self.get_heightmaps(primitive_action,
                        self.action_dict[stack_height]['place_image_ind'], use_hist=use_hist)

        # get stack features if stack_trainer is provided
        # TODO(adit98) can add specific rotation to these forward calls for speedup
        if stack_trainer is not None:
            # to get vector of 64 vals, run trainer.forward with get_action_feat
            stack_push, stack_grasp, stack_place = stack_trainer.forward(color_heightmap,
                    valid_depth_heightmap, is_volatile=True, keep_action_feat=True)[:3]

            # fill all masked arrays (convert to regular np arrays)
            stack_push, stack_grasp, stack_place = stack_push.filled(0.0), \
                    stack_grasp.filled(0.0), stack_place.filled(0.0)

        # get row features if row_trainer is provided
        if row_trainer is not None:
            # to get vector of 64 vals, run trainer.forward with get_action_feat
            row_push, row_grasp, row_place = row_trainer.forward(color_heightmap,
                    valid_depth_heightmap, is_volatile=True, keep_action_feat=True)[:3]

            # fill all masked arrays (convert to regular np arrays)
            row_push, row_grasp, row_place = row_push.filled(0.0), \
                    row_grasp.filled(0.0), row_place.filled(0.0)

        # get unstack features if unstack_trainer is provided
        if unstack_trainer is not None:
            # to get vector of 64 vals, run trainer.forward with get_action_feat
            unstack_push, unstack_grasp, unstack_place = unstack_trainer.forward(color_heightmap,
                    valid_depth_heightmap, is_volatile=True, keep_action_feat=True)[:3]

            # fill all masked arrays (convert to regular np arrays)
            unstack_push, unstack_grasp, unstack_place = unstack_push.filled(0.0), \
                    unstack_grasp.filled(0.0), unstack_place.filled(0.0)

        # get vertical_square features if vertical_square_trainer is provided
        if vertical_square_trainer is not None:
            # to get vector of 64 vals, run trainer.forward with get_action_feat
            vertical_square_push, vertical_square_grasp, vertical_square_place = vertical_square_trainer.forward(color_heightmap,
                    valid_depth_heightmap, is_volatile=True, keep_action_feat=True)[:3]

            # fill all masked arrays (convert to regular np arrays)
            vertical_square_push, vertical_square_grasp, vertical_square_place = vertical_square_push.filled(0.0), \
                    vertical_square_grasp.filled(0.0), vertical_square_place.filled(0.0)

        # get demo action index vector
        action_vec = self.action_dict[stack_height][ACTION_TO_ID[primitive_action]]

        # convert rotation angle to index
        best_rot_ind = np.around((np.rad2deg(action_vec[-2]) % 360) * 16 / 360).astype(int)

        # convert robot coordinates to pixel
        workspace_pixel_offset = workspace_limits[:2, 0] * -1 * 1000
        best_action_xy = ((workspace_pixel_offset + 1000 * action_vec[:2]) / 2).astype(int)

        # initialize best actions for stacking and row making
        best_action_stack, best_action_row, best_action_unstack, best_action_vertical_square = None, None, None, None

        # index predictions to obtain best action
        if primitive_action == 'grasp':
            # NOTE that we swap the order that the best_action_xy coordinates are passed in since
            # the NN output expects (theta, :, y, x)
            if stack_trainer is not None:
                best_action_stack = stack_grasp[best_rot_ind, :, best_action_xy[1],
                        best_action_xy[0]]

            if row_trainer is not None:
                best_action_row = row_grasp[best_rot_ind, :, best_action_xy[1],
                        best_action_xy[0]]

            if unstack_trainer is not None:
                best_action_unstack = unstack_grasp[best_rot_ind, :, best_action_xy[1],
                        best_action_xy[0]]

            if vertical_square_trainer is not None:
                best_action_vertical_square = vertical_square_grasp[best_rot_ind, :,
                        best_action_xy[1], best_action_xy[0]]

        elif primitive_action == 'place':
            if stack_trainer is not None:
                best_action_stack = stack_place[best_rot_ind, :, best_action_xy[1],
                        best_action_xy[0]]

            if row_trainer is not None:
                best_action_row = row_place[best_rot_ind, :, best_action_xy[1],
                        best_action_xy[0]]

            if unstack_trainer is not None:
                best_action_unstack = unstack_place[best_rot_ind, :, best_action_xy[1],
                        best_action_xy[0]]

            if vertical_square_trainer is not None:
                best_action_vertical_square = vertical_square_place[best_rot_ind, :,
                        best_action_xy[1], best_action_xy[0]]

        # return best action for each model, primitive_action
        return best_action_row, best_action_stack, best_action_unstack, best_action_vertical_square, ACTION_TO_ID[primitive_action]
