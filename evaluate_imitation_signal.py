import numpy as np
from scipy import ndimage
import os
import argparse
import cv2

# TODO(adit98) refactor to use ACTION_TO_IND from utils.py
# TODO(adit98) rename im_action.log and im_action_embed.log to be hyphenated

# function to visualize prediction signal on heightmap (with rotations)
def get_prediction_vis(predictions, heightmap, best_pix_ind, scale_factor=8, blend_ratio=0.3):
    canvas = None
    num_rotations = predictions.shape[0]

    # populate canvas
    for canvas_row in range(int(num_rotations/4)):
        tmp_row_canvas = None
        for canvas_col in range(4):
            rotate_idx = canvas_row*4+canvas_col
            prediction_vis = predictions[rotate_idx,:,:].copy()

            # Reduce the dynamic range so the visualization looks better
            prediction_vis = prediction_vis/np.max(prediction_vis)

            # shouldn't be necessary since l2 distances squared are all positive, but just in case
            prediction_vis = np.clip(prediction_vis, 0, 1)

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
                    blend_ratio, prediction_vis, 1-blend_ratio, 0.5)

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
    parser.add_argument('-l', '--log_home', type=str, help='format is logs/EXPERIMENT_DIR')
    parser.add_argument('-v', '--save_visualizations', default=False, action='store_true', help='store depth heightmaps with imitation signal')
    parser.add_argument('-e', '--exec_viz', default=False, action='store_true', help='visualize executed action signal instead of imitation')
    parser.add_argument('-s', '--single_image', default=None, help='visualize signal for only a single image (only works for demo images)')
    args = parser.parse_args()

    # TODO(adit98) may need to make this variable
    # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.5]])

    # we only want to process a single image
    if args.single_image is not None:
        if 'depth' in args.single_image:
            depth_heightmap = cv2.imread(args.single_image, -1)
            rgb_heightmap = cv2.cvtColor(cv2.imread(args.single_image.replace('depth',
                'color')), cv2.COLOR_BGR2RGB)
        elif 'color' in args.single_image:
            depth_heightmap = cv2.imread(args.single_image.replace('color',
                'depth'), -1)
            rgb_heightmap = cv2.cvtColor(cv2.imread(args.single_image),
                    cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("input needs to be either color or depth heightmap")

        # first get the demo num
        demo_num = int(args.single_image.split('/')[-1].split('.')[0])

        # now get the action string e.g. 0grasp, orig, 2place, etc.
        action_str = args.single_image.split('/')[-1].split('.')[1]

        # now get the stack height
        stack_height = action_str[0]
        if stack_height == 'o':
            # string is orig
            stack_height = 0
        else:
            stack_height = int(stack_height)

        # finally get the action ind
        if action_str[1:] == 'grasp':
            action_ind = 2 * stack_height
        elif action_str[1:] == 'place':
            action_ind = 2 * stack_height + 1
        else:
            # if it is 'orig'
            action_ind = 2 * stack_height

        # load best action from logs
        demo_home = "/".join(args.single_image.split('/')[:2])
        action_log = os.path.join(demo_home, 'transitions', "executed-actions-%d.log.txt" % demo_num)
        action_vec = np.loadtxt(action_log)[action_ind][:-1]

        # convert best action from mm to pixels
        best_rot_ind = np.around(np.rad2deg(action_vec[-1]) * 16 / 360).astype(int)
        workspace_pixel_offset = workspace_limits[:2, 0] * -1 * 1000
        best_action_xy = ((workspace_pixel_offset + 1000 * action_vec[:2]) / 2).astype(int)
        best_action_xyt = np.array([best_rot_ind, best_action_xy[1], best_action_xy[0]])

        # generate heatmap
        prob_map = np.ones([16, depth_heightmap.shape[0], depth_heightmap.shape[1]])

        # visualize with rotation, match_ind
        depth_canvas = get_prediction_vis(prob_map, depth_heightmap, best_action_xyt)
        rgb_canvas = get_prediction_vis(prob_map, rgb_heightmap, best_action_xyt)

        # get demo home dir
        if args.save_visualizations:
            # write blended images
            cv2.imwrite(os.path.join(demo_home, 'depth-heightmap.png'), depth_canvas)
            cv2.imwrite(os.path.join(demo_home, 'color-heightmap.png'), rgb_canvas)

    else:
        # make dir for imitation action visualizations if save_visualizations are set
        if args.save_visualizations:
            if args.log_home is None:
                raise ValueError("--log_home is required if not running a single image")

            # we only want files that end in .0 (before action is carried out)
            depth_heightmap_list = sorted([f for f in os.listdir(os.path.join(args.log_home,
                'data', 'depth-heightmaps')) if os.path.isfile(os.path.join(args.log_home,
                    'data', 'depth-heightmaps', f)) and f.endswith("0.depth.png")])
            rgb_heightmap_list = sorted([f for f in os.listdir(os.path.join(args.log_home,
                'data', 'color-heightmaps')) if os.path.isfile(os.path.join(args.log_home,
                    'data', 'color-heightmaps', f)) and f.endswith("0.color.png")])
            print(rgb_heightmap_list)
            depth_home_dir = os.path.join(args.log_home, 'data', 'depth-heightmaps', 'im_depth_signal')
            rgb_home_dir = os.path.join(args.log_home, 'data', 'color-heightmaps', 'im_rgb_signal')
            if not os.path.exists(depth_home_dir):
                os.makedirs(depth_home_dir)
            if not os.path.exists(rgb_home_dir):
                os.makedirs(rgb_home_dir)

        # load action success logs
        grasp_successes = np.loadtxt(os.path.join(args.log_home, 'transitions', 'grasp-success.log.txt'))
        place_successes = np.loadtxt(os.path.join(args.log_home, 'transitions', 'place-success.log.txt'))
        action_success_inds = np.where(np.logical_or(grasp_successes, place_successes))[0]

        # trim array length in case of premature exit
        executed_actions = np.loadtxt(os.path.join(args.log_home, 'transitions', 'executed-action.log.txt'))[:grasp_successes.shape[0]]
        im_actions = np.loadtxt(os.path.join(args.log_home, 'transitions', 'im_action.log.txt'))[:grasp_successes.shape[0]]

        # load imitation embeddings and executed action embeddings
        imitation_embeddings = np.load(os.path.join(args.log_home, 'transitions',
            'im_action_embed.log.txt.npz'), allow_pickle=True)['arr_0'][:grasp_successes.shape[0]]
        executed_action_embeddings = np.load(os.path.join(args.log_home, 'transitions',
            'executed-action-embed.log.txt.npz'), allow_pickle=True)['arr_0'][:grasp_successes.shape[0]]

        # find nearest neighbor for each imitation embedding
        for frame_ind, embedding in enumerate(executed_action_embeddings):
            l2_dist = np.sum(np.square(embedding - np.expand_dims(imitation_embeddings[frame_ind], axis=(0, 2, 3))), axis=1)
            match_ind = np.unravel_index(np.argmin(l2_dist), l2_dist.shape)

            # TODO(adit98) need to add back action_success_inds array
            # evaluate nearest neighbor distance for successful actions
            #if frame_ind in action_success_inds:
            #    # TODO(adit98) calculate euclidean distance between match_ind and executed_action
            #    print('match_ind:', match_ind)
            #    print('executed_action ind:', executed_actions[frame_ind])

            if args.save_visualizations:
                # invert values of l2_dist so that large values indicate correspondence
                im_mask = (np.max(l2_dist) - l2_dist).astype(np.uint8)

                # load original depth/rgb maps
                orig_depth = cv2.imread(os.path.join(args.log_home, 'data', 'depth-heightmaps',
                    depth_heightmap_list[frame_ind]), -1)
                orig_depth = (255 * (orig_depth / np.max(orig_depth))).astype(np.uint8)
                orig_rgb = cv2.imread(os.path.join(args.log_home, 'data', 'color-heightmaps',
                    rgb_heightmap_list[frame_ind]))
                # TODO(adit98) color conversion happens here then reversed in function above, may want to get rid
                orig_rgb = cv2.cvtColor(orig_rgb, cv2.COLOR_BGR2RGB)

                if args.exec_viz:
                    print("action:", executed_actions[frame_ind], "grasp success:",
                            grasp_successes[frame_ind], "place success:", place_successes[frame_ind],
                            "filename:", rgb_heightmap_list[frame_ind])
                    match_ind = executed_actions[frame_ind][1:]

                # flip coordinates of match ind
                match_ind = (match_ind[0], match_ind[1], match_ind[2])

                # visualize with rotation, match_ind
                depth_canvas = get_prediction_vis(im_mask, orig_depth, match_ind)
                rgb_canvas = get_prediction_vis(im_mask, orig_rgb, match_ind)

                # write blended images
                cv2.imwrite(os.path.join(depth_home_dir, depth_heightmap_list[frame_ind]), depth_canvas)
                cv2.imwrite(os.path.join(rgb_home_dir, rgb_heightmap_list[frame_ind]), rgb_canvas)
