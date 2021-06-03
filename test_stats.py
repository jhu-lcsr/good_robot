# test_stats.py collects trial avg progress reversal/recovery stats
import cv2
import os
import re
import argparse
import numpy as np
from utils import get_prediction_vis

from logger import Logger

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', required=True, help='path to logged run')
    parser.add_argument('-s', dest='start_trial', type=int, action='store', default=0,                                help='Trial to start from, default is 0.')
    parser.add_argument('-t', dest='num_trials', type=int, action='store', default=None,                                help='Number of trials to evaluate from start, default is all trials.')
    parser.add_argument('-v', dest='success_height', type=float, action='store', default=4.0,                                help='Max height (number of task progress steps) for considering a trial successful, default is 4, such as a stack of 4 blocks.')
    parser.add_argument('-e', dest='epsilon', type=float, action='store', default=0.1,                                help='Permissible height error margin, i.e. default .1 will count 3.9 as a full stack of 4 when success_height is 4.')
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

    unstack = re.search('unstack', args.data_dir, re.IGNORECASE)

    # update the reward values for a whole trial, not just recent time steps
    end = int(clearance_log[-1][0])
    start_trial = args.start_trial
    clearance_length = len(clearance_log)
    num_trials = clearance_length - 1 # the code typically collects one more trial than needed
    print('clearance_length (total number of trials saved): ' + str(clearance_length) + ' start trial: ' + str(start_trial))
    if args.num_trials is not None:
        num_trials = min(args.num_trials, clearance_length - start_trial)
    epsilon = args.epsilon
    success_height = args.success_height
    print('num_trials (total number of trials to evaluate): ' + str(num_trials))


    max_heights = []
    progress_reversals = []
    recoveries = []
    successful_trials = []
    trial_start = 0
    efficiency_actions_six = 0
    efficiency_actions_four = 0
    for i in range(start_trial, start_trial + num_trials):
        trial_successful = 0
        trial_end = clearance_log[i][0]
        progress_reversal = 0.
        stack_height = 1
        print('----------\ntrial num: ' + str(i))
        print('start: ' + str(trial_start) + ' trial end: ' + str(trial_end ))
        if trial_start == trial_end:
            print('TRIAL ' + str(i) + ' IS EMPTY, skipping')
            continue
        trial_heights = np.array(stack_height_log[trial_start: trial_end])
        print(trial_heights)
        if unstack and len(trial_heights) == 1:
            # special case for unstacking task where the trial ends immediately
            progress_reversal = 1.
            max_heights += [1]
        else:
            max_heights += [np.max(trial_heights)]
        
        # workaround trials that end in fewer actions than it would be possible to optimally succeed
        efficiency_actions_six += max(len(trial_heights), 6)
        efficiency_actions_four += max(len(trial_heights), 4)

        for j in range(trial_start, max(trial_start, trial_end - 1)):
            # allow a little bit of leeway, 0.1 progress, to consider something a reversal
            if stack_height_log[j][0] - epsilon > stack_height_log[j+1][0]:
                progress_reversal = 1.
                print('trial ' + str(i) + ' progress reversal at overall step: ' + str(j) + ' (this trial step: ' + str(j - trial_start) + ') because ' + 
                    str(stack_height_log[j][0] ) + ' - ' + str(epsilon) + ' > ' + str(stack_height_log[j+1][0]))
        progress_reversals += [progress_reversal]
        trial_successful = trial_success_log[trial_end] > trial_success_log[trial_start]
        print('log indicates trial success') if trial_successful else print('log indicates trial failure')
        if unstack and not trial_successful and len(trial_heights) > 1 and max_heights[-1] == 4.:
            # special case for unstack failures, sometimes progress of 4 gets logged after a topple
            print('unstack correction max height: ' + str(max_heights[-1]) + ' to ' + str(trial_heights[-2]))
            max_heights[-1] = float(trial_heights[-2])
        successful_trials += [trial_successful]
        if progress_reversal == 1.:
            recoveries += [1.] if trial_successful == 1 else [0.]
        trial_start = trial_end

    max_heights = np.array(max_heights)
    print('------------------------------')
    print('data dir: ' + str(args.data_dir))
    print('max_heights, the highest height in each trial: ' + str(max_heights))
    print('logged trial success in trial_success_log.txt: ' + str(successful_trials))
    print('trials (success_height - epsilon) height or higher, another way of counting trial successes: ' + str(max_heights >= (success_height - epsilon)))
    print('reversals: ' + str(progress_reversals))
    print('recoveries: ' + str(recoveries))
    print('total trials: ' + str(clearance_length) + ' (clearance_length, total number of trials)')
    print('num trials evaluated: ' + str(num_trials) + ' start trial: ' + str(start_trial))
    print('avg max height: ' + str(np.mean(max_heights)) + ' (higher is better, find max height for each trial, then average those values)')
    print('avg max progress: ' + str(np.mean(np.rint(max_heights))/success_height) + ' (higher is better,  (avg(round(max_heights))/' + str(success_height) + '))')
    print('avg reversals: ' + str(np.mean(progress_reversals)) + ' (lower is better)')
    print('avg recoveries: ' + str(np.mean(recoveries)) + ' (higher is better, no need for recovery attempts is best)')
    print('avg logged trial success: ' + str(np.mean(successful_trials)) + " (successful trials according to trial_success_log.txt)")
    print('avg trial success: ' + str(np.mean(max_heights >= (success_height - epsilon))) + ' (higher is better,  (success_height - epsilon) height or higher)')
    print('action efficiency with 6 action per trial optimum: ' + str((6.*num_trials)/efficiency_actions_six) + ' action efficiency with 4 action per trial optimum: ' + str((4.*num_trials)/efficiency_actions_four))
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
