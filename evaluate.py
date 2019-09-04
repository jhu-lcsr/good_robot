#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Parse session directories
parser = argparse.ArgumentParser(description='Plot performance of a session over training time.')
parser.add_argument('--session_directory', dest='session_directory', action='store', type=str, help='path to session directory for which to measure performance')
parser.add_argument('--method', dest='method', action='store', type=str, help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
parser.add_argument('--num_obj_complete', dest='num_obj_complete', action='store', type=int, help='number of objects picked before considering task complete')
parser.add_argument('--preset', dest='preset', action='store_true', default=False, help='use the 11 preset object locations for difficult situations')
parser.add_argument('--preset_num_trials', dest='preset_num_trials', action='store', type=int, default=10, help='How many trials did you run each preset situation for? (default is 10)')

args = parser.parse_args()
session_directory = args.session_directory
method = args.method
num_obj_complete = args.num_obj_complete
# There is a preset directory with this number of objects each time
num_obj_presets = [4, 5, 3, 5, 5, 6, 3, 6, 6, 5, 4]
# num_obj_presets = [3, 5, 6, 3, 4, 6, 5, 5, 5, 4, 6]
# number of objects per preset case file:
# file: num_obj
# 00: 4
# 01: 5
# 02: 3
# 03: 5
# 04: 5
# 05: 6
# 06: 3
# 07: 6
# 08: 6
# 09: 5
# 10: 4

preset_trials_per_case = args.preset_num_trials

# Parse data from session (action executed, reward values)
# NOTE: reward_value_log just stores some value which is indicative of successful grasping, which could be a class ID (reactive) or actual reward value (from MDP, reinforcement)
transitions_directory = os.path.join(session_directory, 'transitions')
executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
max_iteration = executed_action_log.shape[0]
executed_action_log = executed_action_log[0:max_iteration,:]
reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
grasp_success_log = np.loadtxt(os.path.join(transitions_directory, 'grasp-success.log.txt'), delimiter=' ')
reward_value_log = reward_value_log[0:max_iteration]
clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
# work around a bug where the clearance steps were written twice per clearance
clearance_log = np.unique(clearance_log)
max_trials = len(clearance_log)
clearance_log = np.concatenate((np.asarray([0]), clearance_log), axis=0).astype(int)
print('number of clearances: ' + str(len(clearance_log)-1))
# Count number of pushing/grasping actions before completion
num_actions_before_completion = clearance_log[1:(max_trials+1)] - clearance_log[0:(max_trials)]

grasp_success_rate = np.zeros((max_trials))
grasp_num_success = np.zeros((max_trials))
grasp_to_push_ratio = np.zeros(max_trials)
valid_clearance = []
for trial_idx in range(1, len(clearance_log)):
    if args.preset:
        # TODO(ahundt) If a bug is fixed in the logging code, there is one too many trials here.
        num_preset_files = 11
        preset_situation_num = min(num_preset_files-1, int(float(trial_idx-1)/float(preset_trials_per_case)))
        # preset_situation_num = (trial_idx-1) % num_preset_files
        # preset_situation_num = int((trial_idx-2)/preset_trials_per_case)
        num_obj_complete = num_obj_presets[preset_situation_num]
    # Get actions and reward values for current trial
    start_idx = clearance_log[trial_idx-1]
    end_idx = clearance_log[trial_idx]
    # print('trial: ' + str(trial_idx) + ' start: ' + str(start_idx) + ' end: ' + str(end_idx))
    tmp_executed_action_log = executed_action_log[start_idx:end_idx,0]
    tmp_reward_value_log = reward_value_log[start_idx:end_idx]
    tmp_grasp_success_log = grasp_success_log[start_idx:end_idx]

    # Get indices of pushing and grasping actions for current trial
    tmp_grasp_attempt_ind = np.argwhere(tmp_executed_action_log == 1)
    tmp_push_attempt_ind = np.argwhere(tmp_executed_action_log == 0)

    # print('debug len(tmp_executed_action_log)' + str(len(tmp_executed_action_log)))
    grasp_to_push_ratio[trial_idx-1] = float(len(tmp_grasp_attempt_ind))/float(len(tmp_executed_action_log))

    # Count number of times grasp attempts were successful
    if method == 'reactive':
        tmp_num_grasp_success = np.sum(tmp_reward_value_log[tmp_grasp_attempt_ind] == 0) # Class ID for successful grasping is 0 (reactive policy)
    elif method == 'reinforcement':
        tmp_num_grasp_success = np.sum(tmp_reward_value_log[tmp_grasp_attempt_ind] >= 0.5) # Reward value for successful grasping is anything larger than 0.5 (reinforcement policy)
        # print('trial: ' + str(trial_idx) + ' num grasp success:' + str(tmp_num_grasp_success))
    grasp_num_success[trial_idx-1] = np.sum(tmp_grasp_success_log)
    grasp_success_rate[trial_idx-1] = float(tmp_num_grasp_success)/float(len(tmp_grasp_attempt_ind))
    # Did the trial reach task completion?
    cleared = tmp_num_grasp_success >= num_obj_complete
    if args.preset:
        print('trial: ' + str(trial_idx) + ' preset_situation_num: ' + str(preset_situation_num) + ' num_obj: ' + str(num_obj_complete) +
              ' num_grasp_success: ' + str(tmp_num_grasp_success) + ' cleared: ' + str(cleared) + 
              ' start: ' + str(start_idx) + ' end: ' + str(end_idx) + ' actions: ' + str(len(tmp_executed_action_log)))
    valid_clearance.append(cleared)

# Display results
print('Average %% clearance: %2.1f' % (float(np.sum(valid_clearance))/float(max_trials)*100))
print('Average %% grasp success per clearance: %2.1f' % (np.mean(grasp_success_rate[valid_clearance])*100))
print('Average %% action efficiency: %2.1f' % (100*np.mean(np.divide(float(num_obj_complete), num_actions_before_completion[valid_clearance]))))
print('Average grasp to push ratio: %2.1f' % (np.mean(grasp_to_push_ratio[valid_clearance])*100))
