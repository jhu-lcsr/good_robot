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

# TODO(adit98) refactor to use ACTION_TO_IND from utils.py
# TODO(adit98) rename im_action.log and im_action_embed.log to be hyphenated

# function to evaluate l2 distance and generate demo-signal mask
def evaluate_l2_mask(executed_actions, embedding, frame_ind, mask):
    match_ind = executed_actions[frame_ind][1:].astype(int)
    l2_dist = np.sum(np.square(embedding - np.expand_dims(embedding[match_ind[0],
        :, match_ind[1], match_ind[2]], axis=(0, 2, 3))), axis=1)

    # set masked spaces to have max of l2_dist*1.1 distance
    l2_dist[np.multiply(l2_dist, 1 - mask) == 0] = np.max(l2_dist) * 1.1

    # make l2_dist range from 0 to 1
    l2_dist -= np.min(l2_dist)
    l2_dist /= np.max(l2_dist)

    # invert values of l2_dist so that large values indicate correspondence, exponential to increase dynamic range
    im_mask = 1 - l2_dist

    return im_mask

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
    parser.add_argument('-t', '--task_type', default='unstack', help='task type (enter custom as catch-all)')
    parser.add_argument('-s', '--snapshot_file', dest='snapshot_file', action='store', default='', help='snapshot file to load for the model')
    parser.add_argument('-c', '--cpu', action='store_true', default=False, help='force cpu')
    parser.add_argument('-b', '--blend_ratio', default=0.5, type=float, help='how much to weight background vs similarity heatmap')
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

    # TODO(adit98) define trainer
    # Initialize trainer
    trainer = Trainer(method='reinforcement', push_rewards=True, future_reward_discount=0.5,
                      is_testing=True, snapshot_file=args.snapshot_file,
                      force_cpu=args.cpu, goal_condition_len=0, place=True,
                      pretrained=True, flops=False, network='densenet',
                      common_sense=True, show_heightmap=False, place_dilation=0,
                      common_sense_backprop=True, trial_reward='spot',
                      num_dilation=0)

    # iterate through action_dict and visualize example signal on imitation heightmaps
    action_keys = sorted(example_demo.action_dict.keys())
    for k in action_keys:
        for action in ['grasp', 'place']:
            # get action embedding
            example_action, _ = example_demo.get_action(trainer, workspace_limits,
                    action, k)

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

            # set whether place common sense masks should be used
            # TODO(adit98) make this a cmd line argument
            if args.task_type == 'unstack':
                place_common_sense = True
            else:
                place_common_sense = False

            # run forward pass for imitation_demo
            # to get vector of 64 vals, run trainer.forward with get_action_feat
            push_preds, grasp_preds, place_preds = trainer.forward(im_color,
                    im_depth, is_volatile=True, keep_action_feat=True, use_demo=True,
                    demo_mask=True, place_common_sense=place_common_sense)

            if action == 'grasp':
                preds = grasp_preds
            else:
                preds = place_preds

            # reshape example_action to 1 x 64 x 1 x 1
            example_action = np.expand_dims(example_action, (0, 2, 3))

            # store indices of masked spaces (take min so we enforce all 64 values are 0)
            mask = (np.min((preds == np.zeros([1, 64, 1, 1])).astype(int), axis=1) == 1).astype(int)

            # calculate l2 distance between example action embedding and grasp_preds
            l2_dist = np.sum(np.square(example_action - preds), axis=1)

            # set masked spaces to have max of l2_dist*1.1 distance
            l2_dist[np.multiply(l2_dist, 1 - mask) == 0] = np.max(l2_dist) * 1.1
            match_ind = np.unravel_index(np.argmin(l2_dist), l2_dist.shape)


            if args.save_visualizations:
                # make l2_dist range from 0 to 1
                l2_dist = l2_dist - np.min(l2_dist)
                l2_dist /= np.max(l2_dist)

                # invert values of l2_dist so that large values indicate correspondence, exponential to increase dynamic range
                im_mask = 1 - l2_dist

                # fix dynamic range of im_depth
                im_depth = (im_depth * 255 / np.max(im_depth)).astype(np.uint8)

                ## load original depth/rgb maps
                #orig_depth = cv2.imread(os.path.join(log_home, 'data', 'depth-heightmaps',
                #    depth_heightmap_list[frame_ind]), -1)
                #orig_depth = (255 * (orig_depth / np.max(orig_depth))).astype(np.uint8)
                #orig_rgb = cv2.imread(os.path.join(log_home, 'data', 'color-heightmaps',
                #    rgb_heightmap_list[frame_ind]))
                ## TODO(adit98) color conversion happens here then reversed in function above, may want to get rid
                #orig_rgb = cv2.cvtColor(orig_rgb, cv2.COLOR_BGR2RGB)

                # visualize with rotation, match_ind
                #cv2.imwrite('test.png', im_color)
                #cv2.imwrite('test_depth.png', im_depth)
                depth_canvas = get_prediction_vis(im_mask, im_depth, match_ind, blend_ratio=args.blend_ratio)
                rgb_canvas = get_prediction_vis(im_mask, im_color, match_ind, blend_ratio=args.blend_ratio)

                # write blended images
                cv2.imwrite(depth_filename, depth_canvas)
                cv2.imwrite(color_filename, rgb_canvas)
