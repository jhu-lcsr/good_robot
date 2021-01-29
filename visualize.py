import cv2
from utils import get_prediction_vis
import numpy as np

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', required=True, help='path to logged run')

    # load executed actions
    action_log = np.loadtxt(os.path.join(args.data_dir, 'transitions', 'executed_action.log.txt'))

    # get all heightmap paths
    heightmap_paths = os.listdir(os.path.join(args.data_dir, 'data', 'depth-heightmaps'))

    # filter out the initially saved heightmaps and get the full path
    heightmaps = [os.path.join(args.data_dir, 'data', 'depth-heightmaps', h) \
            for h in heightmaps if '0.depth' in h]

    # iterate through heightmaps in data/depth_heightmaps
    for ind, h in enumerate(heightmaps):
        # load img
        img = cv2.imread(h)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # visualize best pix ind
        img_viz = get_prediction_vis(np.ones_like(img), img, action_log[ind])

        # write img_viz (use same img name as heightmap name)
        cv2.imwrite(os.path.join(args.data_dir, 'visualizations', h.split('/')[-1]))
