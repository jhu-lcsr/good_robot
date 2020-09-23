import numpy as np
from scipy import ndimage
import os
import argparse
import cv2

# TODO(adit98) refactor to use ACTION_TO_IND from utils.py
# TODO(adit98) rename im_action.log and im_action_embed.log to be hyphenated

# function to visualize prediction signal on heightmap (with rotations)
def get_prediction_vis(self, predictions, heightmap, best_pix_ind, scale_factor=8):
    # predictions is a matrix of shape 16x224x224 with l2 distances squared
    print(predictions.shape)
    print(best_pix_ind)
    canvas = None
    num_rotations = predictions.shape[0]
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
                prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (221,211,238), 2)

            # rotate probability map and image to gripper rotation
            prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0).astype(np.uint8)
            background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0).astype(np.uint8)

            # blend image and colorized probability heatmap
            prediction_vis = cv2.addWeighted(cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR), 0.5, prediction_vis, 0.5)

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
    parser.add_argument('-l', '--log_home', required=True, type=str, help='format is logs/EXPERIMENT_DIR')
    parser.add_argument('-v', '--save_visualizations', default=False, action='store_true', help='store depth heightmaps with imitation signal')
    args = parser.parse_args()

    # make dir for imitation action visualizations if save_visualizations are set
    if args.save_visualizations:
        depth_heightmap_list = sorted([f for f in os.listdir(os.path.join(args.log_home,
            'data', 'depth-heightmaps')) if os.path.isfile(os.path.join(args.log_home,
                'data', 'depth-heightmaps', f))])
        rgb_heightmap_list = sorted([f for f in os.listdir(os.path.join(args.log_home,
            'data', 'color-heightmaps')) if os.path.isfile(os.path.join(args.log_home,
                'data', 'color-heightmaps', f))])
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
    imitation_embeddings = np.load(os.path.join(args.log_home, 'transitions', 'im_action_embed.log.txt.npz'), allow_pickle=True)['arr_0']
    executed_action_embeddings = np.load(os.path.join(args.log_home, 'transitions', 'executed-action-embed.log.txt.npz'), allow_pickle=True)['arr_0']

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
            # TODO(adit98) think about this and resolve
            # for now, take min along rotation axis)
            # im_mask = np.min(l2_dist, axis=0)
            # for now, just use unrotated image
            im_mask = l2_dist[0]
            # invert values so that large values indicate correspondence
            im_mask = (255 * (1 - (im_mask / np.max(im_mask)))).astype(np.uint8)
            # apply colormap jet
            im_mask = cv2.applyColorMap(im_mask, cv2.COLORMAP_JET)

            # load original depth/rgb maps
            orig_depth = cv2.imread(os.path.join(args.log_home, 'data', 'depth-heightmaps',
                depth_heightmap_list[frame_ind]), -1)
            orig_depth = (255 * (orig_depth / np.max(orig_depth))).astype(np.uint8)
            orig_rgb = cv2.imread(os.path.join(args.log_home, 'data', 'color-heightmaps',
                rgb_heightmap_list[frame_ind]))

            # TODO(adit98) color conversion happens here then reversed in function above, may want to get rid
            orig_rgb = cv2.cvtColor(orig_rgb, cv2.COLOR_BGR2RGB)

            # visualize with rotation, match_ind
            rgb_canvas = get_prediction_vis(self, l2_dist, orig_rgb, match_ind)
            depth_canvas = get_prediction_vis(self, l2_dist, orig_depth, match_ind)

            # write blended images
            cv2.imwrite(os.path.join(depth_home_dir, depth_heightmap_list[frame_ind]), depth_blended)
            cv2.imwrite(os.path.join(rgb_home_dir, rgb_heightmap_list[frame_ind]), rgb_blended)

            # TODO(adit98) testing, so have break
            break
