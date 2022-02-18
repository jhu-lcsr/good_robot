"""
Generates cycle-consistent distance figure
added by Ran Liu (rliu14@jhu.edu)
though I mostly just copy-pasted code and refactorized some of it
May 19, 2021
"""

import argparse
import cv2
import matplotlib.pyplot
import numpy
from matplotlib import pyplot as plt

from demo import Demonstration
from trainer import Trainer


def read_image(path: str, img_type: str):
    """
    Reads image into numpy array
    @param path: Path to image
    @param img_type: One of 'color', 'depth'
    @return: Array containing image contents
    """
    # This is repeated several times in the code and should ideally be refactored into a function

    if img_type == "color":
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    elif img_type == "depth":
        return numpy.stack([cv2.imread(path, -1)]*3, axis=-1).astype(numpy.float32)/100000
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-w", "--workspace", type=float, nargs=6,
                        default=[-0.724, -0.276, -0.224, 0.224, -0.0001, 0.5],
                        help="Workspace limits (xmin xmax ymin ymax zmin zmax)")

    parser.add_argument("-s", "--snapshot", type=str, default=None, help="Path to snapshot file.", required=True)
    parser.add_argument("-d", "--demo", type=str, default=None, help="Path to directory of demo.", required=True)
    parser.add_argument("-rc", "--real-color", type=str, default=None, help="Path to real color map.", required=True)
    parser.add_argument("-rd", "--real-depth", type=str, default=None, help="Path to real depth map.", required=True)
    parser.add_argument("-h", "--stack-height", type=int, default=None, help="Stack height of demo.", required=True)
    parser.add_argument("-n", "--demo-number", type=int, default=None, help="Demo number.", required=True)
    parser.add_argument("-a", "--action-type", type=str, default="grasp", help="Action (push, grasp, place).", required=True)
    parser.add_argument("-i", "--action-index", type=int, default=None, help="Index of action taken.", required=True)

    args = parser.parse_args()

    workspace_limits = numpy.asarray([[args.workspace[0], args.workspace[1]],
                                      [args.workspace[2], args.workspace[3]],
                                      [args.workspace[4], args.workspace[5]]])
    workspace_limits = numpy.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.5]])

    # Load model from snapshot
    stack_snapshot_file = 'logs/base_models/best_stack/snapshot.reinforcement_trial_success_rate_best_value.pth'
    stack_trainer = Trainer(method='reinforcement', push_rewards=True, future_reward_discount=0.5,
                            is_testing=True, snapshot_file=stack_snapshot_file,
                            force_cpu=False, goal_condition_len=0, place=True,
                            pretrained=True, flops=False, network='densenet',
                            common_sense=True, place_common_sense=True,
                            show_heightmap=False, place_dilation=0.05,
                            common_sense_backprop=True, trial_reward='spot',
                            num_dilation=0)

    demo = Demonstration("logs/demos/stack_demos", 0, None)
    demo_color_heightmap, demo_depth_heightmap = demo.get_heightmaps("grasp", 2)

    # demo_features = stack_trainer.forward(demo_color_heightmap, demo_depth_heightmap)
    # actions = demo.get_action(workspace_limits, "grasp", 2, stack_trainer)
    stack_push, stack_grasp, stack_place = stack_trainer.forward(demo_color_heightmap, demo_depth_heightmap,
                                                                 is_volatile=True, keep_action_feat=True,
                                                                 demo_mask=True)[:3]
    # fill all masked arrays (convert to regular np arrays)
    stack_push, stack_grasp, stack_place = stack_push.filled(0.0), stack_grasp.filled(0.0), stack_place.filled(0.0)

    # Real features

    # get demo action index vector
    action_vector = demo.action_dict[1][1]

    # convert rotation angle to index
    best_rot_ind = numpy.around((numpy.rad2deg(action_vector[-2]) % 360) * 16 / 360).astype(int)

    # convert robot coordinates to pixel
    workspace_pixel_offset = workspace_limits[:2, 0] * -1 * 1000
    best_action_xy = ((workspace_pixel_offset + 1000 * action_vector[:2]) / 2).astype(int)

    # [rot, :, x, y]
    # Compute rematch distances


