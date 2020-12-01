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

# function to evaluate l2 distance and generate demo-signal mask
def evaluate_l2_mask(preds, example_actions, demo_hist=None, execution_hist=None):
    # TODO(adit98) modify normalize to do L2 normalization (makes more sense for later steps taking inner product similarity)
    # helper function to normalize input along axis 1 (embedding dim)
    def normalize(x, ax=None):
        if ax is None or ax == 1:
            # define axis (1 if >=2 dims, 0 otherwise)
            if len(x.shape) < 2:
                ax = 0
            else:
                ax = 1

            # max normalization, add small constant so that maximum is non-zero
            x_norm = x - np.min(x, axis=ax, keepdims=True) + 0.00001
            x_norm = x_norm / np.max(x_norm, axis=ax, keepdims=True)

        elif ax == 'all':
            x_norm = x - np.min(x)
            x_norm = x_norm / (np.max(x_norm) + 0.00001)

        return x_norm

    # helper function to compute cosine similarity between pixel-wise predictions and single embedding vector
    def cos_sim(pix_preds, best_pred):
        # pix_preds is the pixel-wise embedding array, best_pred is the single template embedding vector
        best_pred = np.expand_dims(best_pred, (0, 2, 3))
        cos_sim = np.multiply(pix_preds, best_pred)
        return cos_sim

    # reshape each example_action to 1 x 64 x 1 x 1
    example_action_row, example_action_stack = example_actions
    if example_action_row is not None:
        example_action_row = np.expand_dims(example_action_row, (0, 2, 3))
    if example_action_stack is not None:
        example_action_stack = np.expand_dims(example_action_stack, (0, 2, 3))

    # parse preds into row/stack
    row_preds, stack_preds = preds

    # TODO(adit98) see whether row mask and stack mask are different
    # recover masked spaces from predictions (locations where all 64 values are 0)
    if stack_preds is not None:
        mask = (stack_preds == np.zeros([1, 64, 1, 1])).all(axis=1)
    else:
        mask = (row_preds == np.zeros([1, 64, 1, 1])).all(axis=1)

    # add the l2 distances from history if history is given
    if demo_hist is not None and execution_hist is not None:
        # initialize l2_dist
        l2_dist = np.zeros_like(mask).astype(float)
        if row_preds is not None:
            # add row and stack l2 distances
            l2_dist += normalize(np.sum(np.square(row_preds - example_action_row), axis=1), ax='all')
        if stack_preds is not None:
            l2_dist += normalize(np.sum(np.square(stack_preds - example_action_stack), axis=1), ax='all')

        # iterate through execution_hist and demo_hist, compute cosine similarities ((e_t * e_(t-1), d_t * d_(t-1)), ...)
        for i in range(len(execution_hist)):
            # get previous optimal actions from execution, demo
            row_exec_action, stack_exec_action = execution_hist[i]
            row_demo_action, stack_demo_action = demo_hist[i]

            # store each similarity mask (16x64x224x224)
            if row_preds is not None:
                # execution similarities
                row_sim_exec = cos_sim(row_preds, row_exec_action)

                # demo similarities
                row_sim_demo = cos_sim(example_action_row, row_demo_action)

                #sim = np.sum(np.square(row_sim_exec - row_sim_demo), axis=1)
                #print('historical l2 dist dynamic range', np.min(sim), np.max(sim), np.mean(sim))
                #sim_norm = normalize(sim, ax='all')
                #print('normalized historical l2 dist dynamic range', np.min(sim_norm), np.max(sim_norm), np.mean(sim_norm))

                # add to l2 dist
                l2_dist += normalize(np.sum(np.square(row_sim_exec - row_sim_demo), axis=1), ax='all')

            if stack_preds is not None:
                # execution similarities
                stack_sim_exec = cos_sim(stack_preds, stack_exec_action)

                # demo similarities
                stack_sim_demo = cos_sim(example_action_stack, stack_demo_action)

                # add to l2 dist
                l2_dist += normalize(np.sum(np.square(stack_sim_exec - stack_sim_demo), axis=1), ax='all')

    else:
        # calculate l2 distance between example action embedding and preds for each policy (row and stack)
        l2_dist = np.zeros_like(mask).astype(float)
        if example_action_row is not None:
            # normalize sum at every pixel, ax='all' normalizes across all axes
            l2_dist += normalize(np.sum(np.square(example_action_row - row_preds), axis=1), ax='all')
        if example_action_stack is not None:
            l2_dist += normalize(np.sum(np.square(example_action_stack - stack_preds), axis=1), ax='all')

    # set masked spaces to have max of l2_dist*1.1 distance
    l2_dist[mask] = np.max(l2_dist) * 1.1
    match_ind = np.unravel_index(np.argmin(l2_dist), l2_dist.shape)

    # make l2_dist range from 0 to 1
    l2_dist = l2_dist - np.min(l2_dist)
    l2_dist = l2_dist / np.max(l2_dist)

    # invert values of l2_dist so that large values indicate correspondence
    im_mask = (1 - l2_dist)

    return im_mask, match_ind

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
            prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)

            # if this is the correct rotation, draw circle on action coord
            if rotate_idx == best_pix_ind[0]:
                # need to flip best_pix_ind row and col since cv2.circle reads this as (x, y)
                prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (221,211,238), 2)

            # rotate probability map and image to gripper rotation
            prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0).astype(np.uint8)
            background_image = ndimage.rotate(heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0).astype(np.uint8)

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
    parser.add_argument('-v', '--save_visualizations', default=False, action='store_true', help='store depth heightmaps with imitation signal')
    parser.add_argument('-m', '--metric', default='l2', help='metric to evaluate similarity between demo and current env embeddings')
    parser.add_argument('-t', '--task_type', default='custom', help='task type')
    parser.add_argument('-s', '--stack_snapshot_file', default=None, help='snapshot file to load for the stacking model')
    parser.add_argument('-r', '--row_snapshot_file', default=None, help='snapshot file to load for row-making model')
    parser.add_argument('-c', '--cpu', action='store_true', default=False, help='force cpu')
    parser.add_argument('-b', '--blend_ratio', default=0.5, type=float, help='how much to weight background vs similarity heatmap')
    parser.add_argument('-k', '--history_len', default=0, type=int, help='how many historical steps to store')
    args = parser.parse_args()

    # TODO(adit98) may need to make this variable
    # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.5]])

    # create viz directory in imitation_demo folder
    if not os.path.exists(os.path.join(args.imitation_demo, 'correspondences')):
        os.makedirs(os.path.join(args.imitation_demo, 'correspondences'))

    # create both demo classes
    example_demo = Demonstration(path=args.example_demo, demo_num=0,
            check_z_height=True, task_type=args.task_type)
    imitation_demo = Demonstration(path=args.imitation_demo, demo_num=0,
            check_z_height=True, task_type=args.task_type)

    # set whether place common sense masks should be used
    # TODO(adit98) make this a cmd line argument and think about whether it should ever be set
    if args.task_type == 'unstack':
        place_common_sense = False
    else:
        place_common_sense = False

    # Initialize trainer(s)
    stack_trainer, row_trainer = None, None

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

    if args.stack_snapshot_file is None and args.row_snapshot_file is None:
        raise ValueError("Must provide one of stack trained model or row trained model")

    if args.history_len != 0:
        # store previous embeddings
        demo_buffer = []
        execution_buffer = []

        # populate buffers (history_len is the number of steps we store)
        for i in range(args.history_len):
            if stack_trainer is not None and row_trainer is not None:
                demo_buffer.append((np.zeros(64), np.zeros(64)))
                execution_buffer.append((np.zeros(64), np.zeros(64)))
            elif row_trainer is not None:
                demo_buffer.append((np.zeros(64), None))
                execution_buffer.append((np.zeros(64), None))
            else:
                demo_buffer.append((None, np.zeros(64)))
                execution_buffer.append((None, np.zeros(64)))

    else:
        demo_buffer, execution_buffer = None, None

    # iterate through action_dict and visualize example signal on imitation heightmaps
    action_keys = sorted(example_demo.action_dict.keys())
    for k in action_keys:
        for action in ['grasp', 'place']:
            # get action embeddings from example demo
            example_action_row, example_action_stack, _ = example_demo.get_action(workspace_limits,
                    action, k, stack_trainer=stack_trainer, row_trainer=row_trainer)

            # get imitation heightmaps
            if args.task_type == 'unstack':
                if action == 'grasp':
                    im_color, im_depth = imitation_demo.get_heightmaps(action,
                            imitation_demo.action_dict[k]['demo_ind'])
                else:
                    im_color, im_depth = imitation_demo.get_heightmaps(action,
                            imitation_demo.action_dict[k]['demo_ind'] + 1)

            else:
                if action == 'grasp':
                    im_color, im_depth = imitation_demo.get_heightmaps(action,
                            imitation_demo.action_dict[k]['grasp_image_ind'])
                else:
                    im_color, im_depth = imitation_demo.get_heightmaps(action,
                            imitation_demo.action_dict[k]['place_image_ind'])

            # create filenames to be saved
            depth_filename = os.path.join(args.imitation_demo, 'correspondences',
                    str(k) + '.' + action + '.depth.png')
            color_filename = os.path.join(args.imitation_demo, 'correspondences',
                    str(k) + '.' + action + '.color.png')

            # run forward pass for imitation_demo
            stack_preds, row_preds = None, None

            # get stack features if stack_trainer is provided
            if stack_trainer is not None:
                # to get vector of 64 vals, run trainer.forward with get_action_feat
                stack_push, stack_grasp, stack_place = stack_trainer.forward(im_color, im_depth,
                        is_volatile=True, keep_action_feat=True, use_demo=True, demo_mask=True)

            # get row features if row_trainer is provided
            if row_trainer is not None:
                # to get vector of 64 vals, run trainer.forward with get_action_feat
                row_push, row_grasp, row_place = row_trainer.forward(im_color, im_depth,
                        is_volatile=True, keep_action_feat=True, use_demo=True, demo_mask=True)

            # TODO(adit98) add logic for pushing here
            if action == 'grasp':
                if stack_trainer is not None:
                    stack_preds = stack_grasp.filled(0.0)
                if row_trainer is not None:
                    row_preds = row_grasp.filled(0.0)

            else:
                if stack_trainer is not None:
                    stack_preds = stack_place.filled(0.0)
                if row_trainer is not None:
                    row_preds = row_place.filled(0.0)

            # evaluate l2 distance based action mask
            im_mask, match_ind = evaluate_l2_mask(preds=[row_preds, stack_preds],
                    example_actions=[example_action_row, example_action_stack],
                    demo_hist=demo_buffer, execution_hist=execution_buffer)

            if args.save_visualizations:
                # fix dynamic range of im_depth
                im_depth = (im_depth * 255 / np.max(im_depth)).astype(np.uint8)

                # visualize with rotation, match_ind
                depth_canvas = get_prediction_vis(im_mask, im_depth, match_ind, blend_ratio=args.blend_ratio)
                rgb_canvas = get_prediction_vis(im_mask, im_color, match_ind, blend_ratio=args.blend_ratio)

                # write blended images
                cv2.imwrite(depth_filename, depth_canvas)
                cv2.imwrite(color_filename, rgb_canvas)

            if args.history_len != 0:
                # update buffers (add current action, delete first element in buffer)
                demo_buffer.append((example_action_row, example_action_stack))
                del demo_buffer[0]

                # check if row/stack preds are available first because we need to index them if they are
                if row_preds is not None and stack_preds is not None:
                    execution_buffer.append((row_preds[match_ind[0], :, match_ind[1], match_ind[2]],
                        stack_preds[match_ind[0], :, match_ind[1], match_ind[2]]))
                elif row_preds is not None:
                    execution_buffer.append((row_preds[match_ind[0], :, match_ind[1], match_ind[2]], None))
                else:
                    execution_buffer.append((None, stack_preds[match_ind[0], :, match_ind[1], match_ind[2]]))

                del execution_buffer[0]