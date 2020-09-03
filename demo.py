import numpy as np
import cv2
import os

class Demonstration():
    def __init__(self, path, demo_num):
        # path is expected to be <logs/exp_name>
        self.action_log = np.loadtxt(os.path.join(path, 'transitions',
            'executed-actions-0.log.txt'))
        self.rgb_dir = os.path.join(path, 'data', 'color-heightmaps')
        self.depth_dir = os.path.join(path, 'data', 'depth-heightmaps')
        self.demo_num = demo_num

        # these vars keep track of which heightmaps to load
        self.successful_stacks = 0
        self.action_str = 'orig'

        # keep track of which action we are on
        self.frame = 0

    def get_heightmaps(self):
        # e.g. initial rgb filename is 000000.orig.color.png
        rgb_filename = os.path.join(self.rgb_dir, 
                '%06d.%s.color.png' % (self.demo_num, self.action_str))
        depth_filename = os.path.join(self.depth_dir,
                '%06d.%s.depth.png' % (self.demo_num, self.action_str))

        rgb_heightmap = cv2.cvtColor(cv2.imread(rgb_filename), cv2.COLOR_BGR2RGB)
        depth_heightmap = cv2.imread(depth_filename, -1).astype(np.float32)/100000

        return rgb_heightmap, depth_heightmap

    # TODO(adit98) figure out where to get workspace limits
    def get_action(self, trainer, nonlocal_variables, workspace_limits):
        color_heightmap, valid_depth_heightmap = self.get_heightmaps()
        # to get vector of 64 vals, run trainer.forward with get_action_feat
        push_preds, grasp_preds, place_preds, state_feat, output_prob = trainer.forward(color_heightmap,
                valid_depth_heightmap, is_volatile=True, keep_action_feat=True)

        # TODO(adit98) figure out how to convert rotation angle to index
        best_rot_ind = np.around(np.rad2deg(self.action_log[self.frame][-2]) * 16 / 360).astype(int)

        # TODO(adit98) convert robot coordinates to pixel
        workspace_pixel_offset = workspace_limits[:2, 0] * -1 * 1000
        best_action_xy = ((workspace_pixel_offset + 1000 * self.action_log[self.frame][:2]) / 2).astype(int)

        # TODO(adit98) figure out if we want more than 1 coordinate
        if self.action_log[self.frame][-1] == 0:
            # demo is grasp
            # TODO(adit98) check format of grasp_preds and action_log[self.frame][:-1]
            print('original action log:', self.action_log[self.frame])
            print('selected rotation:', best_rot_ind)
            print('selected xy coord:', best_action_xy)
            print('grasp preds shape:', grasp_preds.shape)
            best_action = grasp_preds[best_rot_ind, :, best_action_xy[0], best_action_xy[1]]
            print('best action shape:', best_action.shape)

        # TODO(adit98) add logic for pushing here
        elif self.action_log[self.frame][-1] == 1:
            # demo is push
            raise NotImplementedError

        elif self.action_log[self.frame][-1] == 2:
            # demo is place
            best_action = place_preds[self.action_log[self.frame][:-1].astype(int)]

        return best_action

    # only gets called if an action is successful at test time
    def next(self):
        if 'place' in self.action_str:
            self.successful_stacks += 1
            self.action_str = str(self.successful_stacks) + 'grasp'
        else:
            self.action_str = str(self.successful_stacks) + 'place'

        self.frame += 1
