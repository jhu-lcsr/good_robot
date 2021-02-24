import numpy as np
from scipy import ndimage
import os
import argparse
import cv2
import torch
from collections import OrderedDict
from utils import ACTION_TO_ID, compute_demo_dist, get_prediction_vis, compute_cc_dist
from trainer import Trainer
from demo import Demonstration, load_all_demos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--example_demo', type=str, help='path to example demo')
    parser.add_argument('-d', '--imitation_demo', type=str, help='path to imitation demo')
    parser.add_argument('-m', '--metric', default='l2', help='metric to evaluate similarity between demo and current env embeddings')
    parser.add_argument('-t', '--task_type', default='custom', help='task type')
    parser.add_argument('-s', '--stack_snapshot_file', default=None, help='snapshot file to load for the stacking model')
    parser.add_argument('-r', '--row_snapshot_file', default=None, help='snapshot file to load for row model')
    parser.add_argument('-u', '--unstack_snapshot_file', default=None, help='snapshot file to load for unstacking model')
    parser.add_argument('-v', '--vertical_square_snapshot_file', default=None, help='snapshot file to load for vertical_square model')
    parser.add_argument('-c', '--cpu', action='store_true', default=False, help='force cpu')
    parser.add_argument('-b', '--blend_ratio', default=0.5, type=float, help='how much to weight background vs similarity heatmap')
    parser.add_argument('--cycle_consistency', default=False, action='store_true', help='use cycle consistency to get matching action in demo')
    parser.add_argument('--depth_channels_history', default=False, action='store_true', help='use depth channel history when passing frames to model?')
    parser.add_argument('--viz', dest='save_visualizations', default=False, action='store_true', help='store depth heightmaps with imitation signal')

    args = parser.parse_args()

    # TODO(adit98) may need to make this variable
    # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.5]])

    # create viz directory in imitation_demo folder
    if args.save_visualizations:
        if not os.path.exists(os.path.join(args.imitation_demo, 'correspondences')):
            os.makedirs(os.path.join(args.imitation_demo, 'correspondences'))

    # create both demo classes
    example_demos = load_all_demos(demo_path=args.example_demo, check_z_height=False,
            task_type=args.task_type)
    imitation_demo = Demonstration(path=args.imitation_demo, demo_num=0,
            check_z_height=False, task_type=args.task_type)

    # set whether place common sense masks should be used
    # TODO(adit98) make this a cmd line argument and think about whether it should ever be set
    if args.task_type == 'unstack':
        place_common_sense = False
        demo_mask = True
    else:
        place_common_sense = True
        demo_mask = True

    # Initialize trainer(s)
    stack_trainer, row_trainer, unstack_trainer, vertical_square_trainer = None, None, None, None

    # load stacking if provided
    if args.stack_snapshot_file is not None:
        stack_trainer = Trainer(method='reinforcement', push_rewards=True, future_reward_discount=0.5,
                          is_testing=True, snapshot_file=args.stack_snapshot_file,
                          force_cpu=args.cpu, goal_condition_len=0, place=True,
                          pretrained=True, flops=False, network='densenet',
                          common_sense=True, place_common_sense=place_common_sense,
                          show_heightmap=False, place_dilation=0.05,
                          common_sense_backprop=True, trial_reward='spot',
                          num_dilation=0)

    # load row making if provided
    if args.row_snapshot_file is not None:
        row_trainer = Trainer(method='reinforcement', push_rewards=True, future_reward_discount=0.5,
                          is_testing=True, snapshot_file=args.row_snapshot_file,
                          force_cpu=args.cpu, goal_condition_len=0, place=True,
                          pretrained=True, flops=False, network='densenet',
                          common_sense=True, place_common_sense=place_common_sense,
                          show_heightmap=False, place_dilation=0.05,
                          common_sense_backprop=True, trial_reward='spot',
                          num_dilation=0)

    # load unstack making if provided
    if args.unstack_snapshot_file is not None:
        unstack_trainer = Trainer(method='reinforcement', push_rewards=True, future_reward_discount=0.5,
                          is_testing=True, snapshot_file=args.unstack_snapshot_file,
                          force_cpu=args.cpu, goal_condition_len=0, place=True,
                          pretrained=True, flops=False, network='densenet',
                          common_sense=True, place_common_sense=place_common_sense,
                          show_heightmap=False, place_dilation=0.05,
                          common_sense_backprop=True, trial_reward='spot',
                          num_dilation=0)

    # load vertical_square making if provided
    if args.vertical_square_snapshot_file is not None:
        vertical_square_trainer = Trainer(method='reinforcement', push_rewards=True, future_reward_discount=0.5,
                          is_testing=True, snapshot_file=args.vertical_square_snapshot_file,
                          force_cpu=args.cpu, goal_condition_len=0, place=True,
                          pretrained=True, flops=False, network='densenet',
                          common_sense=True, place_common_sense=place_common_sense,
                          show_heightmap=False, place_dilation=0.05,
                          common_sense_backprop=True, trial_reward='spot', num_dilation=0)

    if stack_trainer is None and row_trainer is None and unstack_trainer is None and vertical_square_trainer is None:
        raise ValueError("Must provide at least one trained model")

    # iterate through action_dict and visualize example signal on imitation heightmaps
    action_keys = sorted(example_demos[0].action_dict.keys())
    example_actions_dict = {}
    for k in action_keys:
        if k not in example_actions_dict:
            example_actions_dict[k] = {}

        for action in ['grasp', 'place']:
            if action not in example_actions_dict[k]:
                example_actions_dict[k][action] = {}

            for ind, d in enumerate(example_demos):
                # get action embeddings from example demo
                if ind not in example_actions_dict[k][action]:
                    example_action_row, example_action_stack, example_action_unstack, example_action_vertical_square, _, demo_action_ind = \
                            d.get_action(workspace_limits, action, k, stack_trainer=stack_trainer, row_trainer=row_trainer,
                                    unstack_trainer=unstack_trainer, vertical_square_trainer=vertical_square_trainer,
                                    use_hist=args.depth_channels_history, demo_mask=True,
                                    cycle_consistency=args.cycle_consistency)
                    example_actions_dict[k][action][ind] = [example_action_row, example_action_stack,
                            example_action_unstack, example_action_vertical_square, demo_action_ind]

            if action == 'grasp':
                im_color, im_depth = imitation_demo.get_heightmaps(action,
                        imitation_demo.action_dict[k]['grasp_image_ind'], use_hist=args.depth_channels_history)
            else:
                im_color, im_depth = imitation_demo.get_heightmaps(action,
                        imitation_demo.action_dict[k]['place_image_ind'], use_hist=args.depth_channels_history)

            # create filenames to be saved
            depth_filename = os.path.join(args.imitation_demo, 'correspondences',
                    str(k) + '.' + action + '.depth.png')
            color_filename = os.path.join(args.imitation_demo, 'correspondences',
                    str(k) + '.' + action + '.color.png')

            # run forward pass for imitation_demo
            stack_preds, row_preds, unstack_preds, vertical_square_preds = None, None, None, None

            # get stack features if stack_trainer is provided
            if stack_trainer is not None:
                # to get vector of 64 vals, run trainer.forward with keep_action_feat
                stack_push, stack_grasp, stack_place = stack_trainer.forward(im_color,
                        im_depth, is_volatile=True, keep_action_feat=True, demo_mask=True)[:3]

                # fill all masked arrays (convert to regular np arrays)
                stack_push, stack_grasp, stack_place = stack_push.filled(0.0), \
                        stack_grasp.filled(0.0), stack_place.filled(0.0)

            # get row features if row_trainer is provided
            if row_trainer is not None:
                # to get vector of 64 vals, run trainer.forward with keep_action_feat
                row_push, row_grasp, row_place = row_trainer.forward(im_color,
                        im_depth, is_volatile=True, keep_action_feat=True, demo_mask=True)[:3]

                # fill all masked arrays (convert to regular np arrays)
                row_push, row_grasp, row_place = row_push.filled(0.0), \
                        row_grasp.filled(0.0), row_place.filled(0.0)

            # get unstack features if unstack_trainer is provided
            if unstack_trainer is not None:
                # to get vector of 64 vals, run trainer.forward with keep_action_feat
                unstack_push, unstack_grasp, unstack_place = unstack_trainer.forward(im_color,
                        im_depth, is_volatile=True, keep_action_feat=True, demo_mask=True)[:3]

                # fill all masked arrays (convert to regular np arrays)
                unstack_push, unstack_grasp, unstack_place = unstack_push.filled(0.0), \
                        unstack_grasp.filled(0.0), unstack_place.filled(0.0)

            # get vertical_square features if vertical_square_trainer is provided
            if vertical_square_trainer is not None:
                # to get vector of 64 vals, run trainer.forward with keep_action_feat
                vertical_square_push, vertical_square_grasp, vertical_square_place = \
                        vertical_square_trainer.forward(im_color, im_depth,
                                is_volatile=True, keep_action_feat=True, demo_mask=True)[:3]

                # fill all masked arrays (convert to regular np arrays)
                vertical_square_push, vertical_square_grasp, vertical_square_place = \
                        vertical_square_push.filled(0.0), vertical_square_grasp.filled(0.0), vertical_square_place.filled(0.0)

            # TODO(adit98) add logic for pushing here
            if action == 'grasp':
                if stack_trainer is not None:
                    stack_preds = stack_grasp
                if row_trainer is not None:
                    row_preds = row_grasp
                if unstack_trainer is not None:
                    unstack_preds = unstack_grasp
                if vertical_square_trainer is not None:
                    vertical_square_preds = vertical_square_grasp

            else:
                if stack_trainer is not None:
                    stack_preds = stack_place
                if row_trainer is not None:
                    row_preds = row_place
                if unstack_trainer is not None:
                    unstack_preds = unstack_place
                if vertical_square_trainer is not None:
                    vertical_square_preds = vertical_square_place

            print("Evaluating distance for stack height:", k, "| Action:", action)

            # rearrange example actions dictionary into (P, D) array where P is number of policies, D # of demos
            example_actions = np.array([*example_actions_dict[k][action].values()], dtype=object).T

            # extract demo action inds
            demo_action_inds = example_actions[-1].tolist()

            # store preds we want to use (after leave one out) in preds, and get relevant example actions
            # order of example actions is row, stack, unstack, vertical square
            if args.task_type == 'row':
                preds = [stack_preds, unstack_preds, vertical_square_preds]
                example_actions = example_actions[1:-1].tolist()
            elif args.task_type == 'stack':
                preds = [row_preds, unstack_preds, vertical_square_preds]
                example_actions = example_actions[[0, 2, 3]].tolist()
            elif args.task_type == 'unstack':
                preds = [row_preds, stack_preds, vertical_square_preds]
                example_actions = example_actions[[0, 1, 3]].tolist()
            elif args.task_type == 'vertical_square':
                preds = [row_preds, stack_preds, unstack_preds]
                example_actions = example_actions[:3].tolist()
            else:
                raise NotImplementedError(args.task_type + ' is not implemented.')

            if not args.cycle_consistency:
                # evaluate distance based action mask - leave one out is above
                im_mask, match_ind = compute_demo_dist(preds=preds, example_actions=example_actions,
                        metric=args.metric)

            else:
                # evaluate distance based action mask with cycle consistency
                im_mask, match_ind = compute_cc_dist(preds=preds, example_actions=example_actions,
                        demo_action_inds=demo_action_inds, valid_depth_heightmap=im_depth, metric=args.metric)

            if args.save_visualizations:
                # fix dynamic range of im_depth
                im_depth = (im_depth * 255 / np.max(im_depth)).astype(np.uint8)

                # visualize with rotation, match_ind
                depth_canvas = get_prediction_vis(im_mask, im_depth, match_ind, blend_ratio=args.blend_ratio)
                rgb_canvas = get_prediction_vis(im_mask, im_color, match_ind, blend_ratio=args.blend_ratio)

                # write blended images
                cv2.imwrite(depth_filename, depth_canvas)
                cv2.imwrite(color_filename, rgb_canvas)
