import cv2
import os
import argparse
import numpy as np
from utils import get_prediction_vis

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demo_dir', required=True, help='path to logged run')
    args = parser.parse_args()

    # load executed actions
    action_log = np.loadtxt(os.path.join(args.demo_dir, 'transitions', 'executed-actions-0.log.txt'))

    # get all heightmap paths
    heightmap_paths = os.listdir(os.path.join(args.demo_dir, 'data', 'depth-heightmaps'))

    # filter out the initially saved heightmaps and get the full path
    heightmap_paths = sorted([os.path.join(args.demo_dir, 'data', 'depth-heightmaps', h) \
            for h in heightmap_paths])

    # define workspace limits
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.5]])

    # iterate through heightmaps in data/depth_heightmaps
    for ind, h in enumerate(heightmap_paths):
        # make sure we break if we run out of logged actions (in case run ended unexpectedly)
        if ind >= len(action_log):
            break

        # load img
        img = cv2.imread(h)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # load action
        action_vec = action_log[ind]

        # convert rotation angle to index
        best_rot_ind = np.around((np.rad2deg(action_vec[-2]) % 360) * 16 / 360).astype(int)

        # convert robot coordinates to pixel
        workspace_pixel_offset = workspace_limits[:2, 0] * -1 * 1000
        best_action_xy = ((workspace_pixel_offset + 1000 * action_vec[:2]) / 2).astype(int)

        # visualize best pix ind
        img_viz = get_prediction_vis(np.ones_like(img), img, (best_rot_ind, best_action_xy[1],
            best_action_xy[0]), specific_rotation=True, num_rotations=16)

        # write img_viz (use same img name as heightmap name)
        cv2.imwrite(os.path.join(args.demo_dir, 'visualizations', h.split('/')[-1]), img_viz)
