# test_stats.py collects trial avg progress reversal/recovery stats
import cv2
import os
import argparse
import numpy as np
from utils import get_prediction_vis

from logger import Logger

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

    kwargs = {'delimiter': ' ', 'ndmin': 2}
    iteration = int(np.loadtxt(os.path.join(args.data_dir, 'transitions', 'iteration.log.txt'), **kwargs)[0, 0])

    stack_height_log = np.loadtxt(os.path.join(args.data_dir, 'transitions', 'stack-height.log.txt'), **kwargs)
    stack_height_log = stack_height_log[0:iteration]
    stack_height_log = stack_height_log.tolist()
    partial_stack_success_log = np.loadtxt(os.path.join(args.data_dir, 'transitions', 'partial-stack-success.log.txt'), **kwargs)
    partial_stack_success_log = partial_stack_success_log[0:iteration]
    partial_stack_success_log = partial_stack_success_log.tolist()
    place_success_log = np.loadtxt(os.path.join(args.data_dir, 'transitions', 'place-success.log.txt'), **kwargs)
    place_success_log = place_success_log[0:iteration]
    place_success_log = place_success_log.tolist()
    trial_success_log = np.loadtxt(os.path.join(args.data_dir, 'transitions', 'trial-success.log.txt'), **kwargs)
    trial_success_log = trial_success_log[0:iteration]
    trial_success_log = trial_success_log.tolist()
    
    if os.path.exists(os.path.join(args.data_dir, 'transitions', 'clearance.log.txt')):
        clearance_log = np.loadtxt(os.path.join(args.data_dir, 'transitions', 'clearance.log.txt'), **kwargs).astype(np.int64)
        clearance_log = clearance_log.tolist()

    # update the reward values for a whole trial, not just recent time steps
    end = int(clearance_log[-1][0])
    clearance_length = len(clearance_log)
    print('clearance_length: ' + str(clearance_length))

    max_heights = []
    progress_reversals = []
    recoveries = []
    successful_trials = []
    trial_start = 0
    for i in range(clearance_length-1):
        trial_successful = 0
        trial_end = clearance_log[i][0]
        print('trial num: ' + str(i))
        print('start: ' + str(trial_start) + ' trial end: ' + str(trial_end ))
        if trial_start == trial_end:
            print('TRIAL ' + str(i) + ' IS EMPTY, skipping')
            continue
        trial_heights = np.array(stack_height_log[trial_start: trial_end])
        print(trial_heights)
        max_heights += [np.max(trial_heights)]
        stack_height = 1
        progress_reversal = 0.
        for j in range(trial_start, max(trial_start, trial_end - 1)):
            # allow a little bit of leeway, 0.1 progress, to consider something a reversal
            if stack_height_log[j][0] - 0.1 > stack_height_log[j+1][0]:
                progress_reversal = 1.
                print('trial ' + str(i) + ' progress reversal at overall step: ' + str(j) + ' (this trial step: ' + str(j - trial_start) + ') because ' + str(stack_height_log[j][0] ) + ' - 0.1 > ' + str(stack_height_log[j+1][0]))
        progress_reversals += [progress_reversal]
        trial_successful = trial_success_log[trial_end] > trial_success_log[trial_start]
        successful_trials += [trial_successful]
        if progress_reversal == 1.:
            recoveries += [1.] if trial_successful == 1 else [0.]
        trial_start = trial_end

    max_heights = np.array(max_heights)
    print('------------------------------')
    print('data dir: ' + str(args.data_dir))
    print('max_heights: ' + str(max_heights))
    print('trials over 4 height: ' + str(max_heights > 4.))
    print('reversals: ' + str(progress_reversals))
    print('recoveries: ' + str(recoveries))
    print('avg max height: ' + str(np.mean(max_heights)) + ' (higher is better)')
    print('avg max progress (avg(round(max_heights))/4): ' + str(np.mean(np.rint(max_heights))/4.) + ' (higher is better)')
    print('avg reversals: ' + str(np.mean(progress_reversals)) + ' (lower is better)')
    print('avg recoveries : ' + str(np.mean(recoveries)) + ' (higher is better, no need for recovery attempts is best)')
    print('avg trials over 4 height: ' + str(np.mean(max_heights > 4.)) + ' (higher is better)')
    print('data dir: ' + str(args.data_dir))



    # if end <= len(stack_height_log):
    #     # First entry won't be zero...
    #     if clearance_length == 1:
    #         start = 0
    #     else:
    #         start = int(clearance_log[-2][0])

    #     new_log_values = []
    #     future_r = None
    #     # going backwards in time from most recent to oldest step
    #     for i in range(start, end):

    # # iterate through heightmaps in data/depth_heightmaps
    # for ind, h in enumerate(heightmap_paths):
    #     # make sure we break if we run out of logged actions (in case run ended unexpectedly)
    #     if ind >= len(action_log):
    #         break

    #     # load img
    #     img = cv2.imread(h)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #     # visualize best pix ind
    #     img_viz = get_prediction_vis(np.ones_like(img), img, action_log[ind][1:],
    #             specific_rotation=action_log[ind][1], num_rotations=16)

    #     # write img_viz (use same img name as heightmap name)
    #     cv2.imwrite(os.path.join(args.data_dir, 'visualizations', h.split('/')[-1]), img_viz)
