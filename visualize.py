import cv2
import os
import argparse
import numpy as np
from utils import get_prediction_vis

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', required=True, help='path to logged run')
    args = parser.parse_args()

    # load executed actions
    action_log = np.loadtxt(os.path.join(args.data_dir, 'transitions', 'executed-action.log.txt'))

    # get all heightmap paths
    heightmap_paths = os.listdir(os.path.join(args.data_dir, 'data', 'depth-heightmaps'))

    # filter out the initially saved heightmaps and get the full path
    heightmap_paths = sorted([os.path.join(args.data_dir, 'data', 'depth-heightmaps', h) \
            for h in heightmap_paths if '0.depth' in h])

    # iterate through heightmaps in data/depth_heightmaps
    for ind, h in enumerate(heightmap_paths):
        # make sure we break if we run out of logged actions (in case run ended unexpectedly)
        if ind >= len(action_log):
            break

        # load img
        img = cv2.imread(h)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # visualize best pix ind
        img_viz = get_prediction_vis(np.ones_like(img), img, action_log[ind][1:],
                specific_rotation=action_log[ind][1], num_rotations=16)

        # write img_viz (use same img name as heightmap name)
        cv2.imwrite(os.path.join(args.data_dir, 'visualizations', h.split('/')[-1]), img_viz)
