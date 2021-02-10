import numpy as np
from scipy import ndimage
import os
import argparse
import cv2
import torch
from collections import OrderedDict
from utils import ACTION_TO_ID
from trainer import Trainer
from demo import Demonstration

# TODO(adit98) move this to utils and refactor
# function to evaluate l2 distance and generate demo-signal mask
def evaluate_l2_mask(preds, example_actions):
    # TODO(adit98) see if we should use cos_sim instead of l2_distance as low-level distance metric
    # helper function to compute cosine similarity between pixel-wise predictions and single embedding vector
    def cos_sim(pix_preds, best_pred):
        # pix_preds is the pixel-wise embedding array, best_pred is the single template embedding vector
        best_pred = np.expand_dims(best_pred, (0, 2, 3))
        cos_sim = np.multiply(pix_preds, best_pred)
        return cos_sim

    # reshape each example_action to 1 x 64 x 1 x 1
    for i in range(len(example_actions)):
        action = example_actions[i]

        # skip if we didn't evaluate model i on demo frame
        if action is None:
            continue

        # reshape and update list
        example_actions[i] = np.expand_dims(action, (0, 2, 3))

    # get mask from first available model (NOTE(adit98) see if we need a different strategy for this)
    mask = None
    for pred in preds:
        if pred is not None:
            mask = (preds == np.zeros([1, 64, 1, 1])).all(axis=1)

    # ensure that at least one of the preds is not None
    if mask is None:
        raise ValueError("Must provide at least one non-null pixel-wise embedding array")

    # calculate l2 distance between example action embedding and preds for each policy
    l2_dists = []
    for ind, action in enumerate(example_actions):
        dist = np.sum(np.square(action - preds[ind]), axis=1)

        # set all masked spaces to have max l2 distance
        dist[mask] = np.max(dist) * 1.1

        # append to l2_dists list
        l2_dists.append(dist)

    # select best action as min b/w all dists in l2_dists
    l2_dists = np.stack(l2_dists)

    # find overall minimum distance across all policies and get index
    match_ind = np.unravel_index(np.argmin(l2_dists), l2_dists.shape)

    # select distance array for policy which contained minimum distance index
    l2_dist = l2_dists[match_ind[0]]

    # discard first dimension of match_ind to get it in the form (theta, y, x)
    match_ind = match_ind[1:]

    # make l2_dist >=0 and max_normalize
    l2_dist = l2_dist - np.min(l2_dist)
    l2_dist = l2_dist / np.max(l2_dist)

    # invert values of l2_dist so that large values indicate correspondence
    im_mask = 1 - l2_dist

    return im_mask, match_ind

# TODO(adit98) replace this with utils function
# function to visualize prediction signal on heightmap (with rotations)
def get_prediction_vis(predictions, heightmap, best_pix_ind, blend_ratio=0.5, prob_exp=1):
    canvas = None
    num_rotations = predictions.shape[0]

    # clip values <0 or >1
    predictions = np.clip(predictions, 0, 1)

    # apply exponential
    predictions = predictions ** prob_exp

    # populate canvas
    for canvas_row in range(int(num_rotations/4)):
        tmp_row_canvas = None
        for canvas_col in range(4):
            rotate_idx = canvas_row*4+canvas_col
            prediction_vis = predictions[rotate_idx,:,:].copy()

            # reshape to 224x224 (or whatever image size is), and color
            prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
            prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8),
                    cv2.COLORMAP_JET)

            # if this is the correct rotation, draw circle on action coord
            if rotate_idx == best_pix_ind[0]:
                # need to flip best_pix_ind row and col since cv2.circle reads this as (x, y)
                prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]),
                    int(best_pix_ind[1])), 7, (221,211,238), 2)

            # rotate probability map and image to gripper rotation
            prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations),
                    reshape=False, order=0).astype(np.uint8)
            background_image = ndimage.rotate(heightmap, rotate_idx*(360.0/num_rotations),
                    reshape=False, order=0).astype(np.uint8)

            # blend image and colorized probability heatmap
            prediction_vis = cv2.addWeighted(cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR),
                    blend_ratio, prediction_vis, 1-blend_ratio, 0)

            # add image to row canvas
            if tmp_row_canvas is None:
                tmp_row_canvas = prediction_vis
            else:
                tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)

        # add row canvas to overall image canvas
        if canvas is None:
            canvas = tmp_row_canvas
        else:
            canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

    return canvas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--example_demo', type=str, help='path to example demo')
    parser.add_argument('-d', '--imitation_demo', type=str, help='path to imitation demo')
    parser.add_argument('-m', '--metric', default='l2', help='metric to evaluate similarity between demo and current env embeddings')
    parser.add_argument('-t', '--task_type', default='custom', help='task type')
    parser.add_argument('-s', '--stack_snapshot_file', default=None, help='snapshot file to load for the stacking model')
    parser.add_argument('-r', '--row_snapshot_file', default=None, help='snapshot file to load for row model')
    parser.add_argument('-u', '--unstack_snapshot_file', default=None, help='snapshot file to load for unstacking model')
    parser.add_argument('-r', '--vertical_square_snapshot_file', default=None, help='snapshot file to load for vertical_square model')
    parser.add_argument('-c', '--cpu', action='store_true', default=False, help='force cpu')
    parser.add_argument('-b', '--blend_ratio', default=0.5, type=float, help='how much to weight background vs similarity heatmap')
    parser.add_argument('--depth_channels_history', default=False, action='store_true', help='use depth channel history when passing frames to model?')
    parser.add_argument('--viz', dest='save_visualizations', default=False, action='store_true', help='store depth heightmaps with imitation signal')

    # TODO(adit98) may need to make this variable
    # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.5]])

    # create viz directory in imitation_demo folder
    if not os.path.exists(os.path.join(args.imitation_demo, 'correspondences')):
        os.makedirs(os.path.join(args.imitation_demo, 'correspondences'))

    # create both demo classes
    example_demo = Demonstration(path=args.example_demo, demo_num=0,
            check_z_height=False, task_type=args.task_type)
    imitation_demo = Demonstration(path=args.imitation_demo, demo_num=0,
            check_z_height=False, task_type=args.task_type)

    # set whether place common sense masks should be used
    # TODO(adit98) make this a cmd line argument and think about whether it should ever be set
    if args.task_type == 'unstack':
        place_common_sense = False
    else:
        place_common_sense = False

    # Initialize trainer(s)
    stack_trainer, row_trainer, unstack_trainer, vertical_square_trainer = None, None, None, None

    # load stacking if provided
    if args.stack_snapshot_file is not None:
        stack_trainer = Trainer(method='reinforcement', push_rewards=True, future_reward_discount=0.5,
                          is_testing=True, snapshot_file=args.stack_snapshot_file,
                          force_cpu=args.cpu, goal_condition_len=0, place=True,
                          pretrained=True, flops=False, network='densenet',
                          common_sense=True, place_common_sense=place_common_sense,
                          show_heightmap=False, place_dilation=0,
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
    action_keys = sorted(example_demo.action_dict.keys())
    for k in action_keys:
        for action in ['grasp', 'place']:
            # get action embeddings from example demo
            example_action_row, example_action_stack, example_action_unstack, example_action_vertical_square, _ = \
                    example_demo.get_action(workspace_limits, action, k, stack_trainer=stack_trainer, row_trainer=row_trainer,
                            unstack_trainer=unstack_trainer, vertical_square_trainer=vertical_square_trainer,
                            use_hist=args.depth_channels_history)

            # get imitation heightmaps
            if args.task_type == 'unstack':
                if action == 'grasp':
                    im_color, im_depth = imitation_demo.get_heightmaps(action,
                            imitation_demo.action_dict[k]['demo_ind'], use_hist=args.depth_channels_history)
                else:
                    im_color, im_depth = imitation_demo.get_heightmaps(action,
                            imitation_demo.action_dict[k]['demo_ind'] + 1, use_hist=args.depth_channels_history)

            else:
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
            stack_preds, row_preds, unstack_preds, vertical_square_preds = None, None

            # get stack features if stack_trainer is provided
            if stack_trainer is not None:
                # to get vector of 64 vals, run trainer.forward with get_action_feat
                stack_push, stack_grasp, stack_place = stack_trainer.forward(im_color,
                        im_depth, is_volatile=True, keep_action_feat=True, demo_mask=True)[:3]

                # fill all masked arrays (convert to regular np arrays)
                stack_push, stack_grasp, stack_place = stack_push.filled(0.0), \
                        stack_grasp.filled(0.0), stack_place.filled(0.0)

            # get row features if row_trainer is provided
            if row_trainer is not None:
                # to get vector of 64 vals, run trainer.forward with get_action_feat
                row_push, row_grasp, row_place = row_trainer.forward(im_color,
                        im_depth, is_volatile=True, keep_action_feat=True, demo_mask=True)[:3]

                # fill all masked arrays (convert to regular np arrays)
                row_push, row_grasp, row_place = row_push.filled(0.0), \
                        row_grasp.filled(0.0), row_place.filled(0.0)

            # get unstack features if unstack_trainer is provided
            if unstack_trainer is not None:
                # to get vector of 64 vals, run trainer.forward with get_action_feat
                unstack_push, unstack_grasp, unstack_place = unstack_trainer.forward(im_color,
                        im_depth, is_volatile=True, keep_action_feat=True, demo_mask=True)[:3]

                # fill all masked arrays (convert to regular np arrays)
                unstack_push, unstack_grasp, unstack_place = unstack_push.filled(0.0), \
                        unstack_grasp.filled(0.0), unstack_place.filled(0.0)

            # get vertical_square features if vertical_square_trainer is provided
            if vertical_square_trainer is not None:
                # to get vector of 64 vals, run trainer.forward with get_action_feat
                vertical_square_push, vertical_square_grasp, vertical_square_place = \
                        vertical_square_trainer.forward(im_color, im_depth,
                                is_volatile=True, keep_action_feat=True, demo_mask=True)[:3]

                # fill all masked arrays (convert to regular np arrays)
                vertical_square_push, vertical_square_grasp, vertical_square_place = \
                        vertical_square_push.filled(0.0), vertical_square_grasp.filled(0.0),
                        vertical_square_place.filled(0.0)

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

            print("Evaluating l2 distance for stack height:", k)
            elif task_type == 'row':
                preds = [stack_preds, unstack_preds, vertical_square_preds]
                example_actions = [example_action_stack, example_action_unstack, example_action_vertical_square]
            if task_type == 'stack':
                preds = [row_preds, unstack_preds, vertical_square_preds]
                example_actions = [example_action_row, example_action_unstack, example_action_vertical_square]
            elif task_type == 'unstack':
                preds = [row_preds, stack_preds, vertical_square_preds]
                example_actions = [example_action_row, example_action_stack, example_action_vertical_square]
            elif task_type == 'vertical_square':
                preds = [row_preds, stack_preds, unstack_preds]
                example_actions = [example_action_row, example_action_stack, example_action_unstack]
            else:
                raise NotImplementedError(task_type + ' is not implemented.')

            # evaluate l2 distance based action mask - leave one out is above
            im_mask, match_ind = evaluate_l2_mask(preds=preds, example_actions=example_actions)

            if args.save_visualizations:
                # fix dynamic range of im_depth
                im_depth = (im_depth * 255 / np.max(im_depth)).astype(np.uint8)

                # visualize with rotation, match_ind
                depth_canvas = get_prediction_vis(im_mask, im_depth, match_ind, blend_ratio=args.blend_ratio)
                rgb_canvas = get_prediction_vis(im_mask, im_color, match_ind, blend_ratio=args.blend_ratio)

                # write blended images
                cv2.imwrite(depth_filename, depth_canvas)
                cv2.imwrite(color_filename, rgb_canvas)
