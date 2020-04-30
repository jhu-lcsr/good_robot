import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from glob import glob
import utils
import scipy


def best_success_rate(success_rate, window, title):
    # Print the best success rate ever
    dict_title = str(title).replace(' ', '_')
    best_dict = {dict_title + '_best_value': float(-np.inf), dict_title + '_best_index': None}
    if success_rate.shape[0] > window:
        best = np.max(success_rate[window:])
        # If there are multiple entries with equal maximum success rates, take the last one because it will have the most training.
        best_index = np.max(np.argmax(success_rate[window:], axis=0)) + window
        best_dict = {dict_title + '_best_value': float(best), dict_title + '_best_index': int(best_index)}
        print('Max ' + title + ': ' + str(best) +
              ', at action iteration: ' + str(best_index) +
              '. (total of ' + str(success_rate.shape[0]) + ' actions, max excludes first ' + str(window) + ' actions)')
    return best_dict


def count_preset_arrangements(trial_complete_indices, trial_successes, num_preset_arrangements, hotfix_trial_success_index=True, log_dir=None):
    arrangement_successes = np.zeros(num_preset_arrangements)
    trials_per_arrangement = int(float(len(trial_complete_indices)) / float(num_preset_arrangements))
    arrangement_trials = np.array([trials_per_arrangement]*num_preset_arrangements)
    if hotfix_trial_success_index:
        # TODO(ahundt) currently the trial success values are inserted too early in the array. Fix then set hotfix param above to false
        trial_successes = np.insert(trial_successes, [0]*3, 0)
    num_arrangements_complete = 0
    length = np.min([np.max(trial_complete_indices), trial_successes.shape[0]])
    arrangement_idx = 0
    trial_num = 0
    clearance_start = 0
    successes_this_arrangement = 0
    prev_trial_successes = 0
    print('max trial successes: ' + str(np.max(trial_successes)))
    for trial_num, index in enumerate(trial_complete_indices):
        index = int(index)
        cur_trial_successes = np.max(trial_successes[clearance_start:index])
        # print(success)
        arrangement_successes[arrangement_idx] += prev_trial_successes < cur_trial_successes
        prev_trial_successes = cur_trial_successes
        if trial_num > 0 and trial_num % trials_per_arrangement == 0:
            arrangement_idx += 1
        clearance_start = index
    individual_arrangement_trial_success_rate = np.divide(np.array(arrangement_successes), arrangement_trials, out=np.zeros(num_preset_arrangements), where=arrangement_trials!=0.0)
    print('individual_arrangement_trial_success_rate: ' + str(individual_arrangement_trial_success_rate))
    senarios_100_percent_complete = np.sum(individual_arrangement_trial_success_rate == 1.0)
    print('senarios_100_percent_complete: ' + str(senarios_100_percent_complete))
    best_dict = {'senarios_100_percent_complete': senarios_100_percent_complete}
    return best_dict


def get_trial_success_rate(trials, trial_successes, window=200, hotfix_trial_success_index=True):
    """Evaluate moving window of grasp success rate
    trials: Nx1 array of the current total trial count at that action
    trial_successes: Nx1 array of the current total successful trial count at the time of that action

    """
    length = np.min([trials.shape[0], trial_successes.shape[0]])
    success_rate = np.zeros(length - 1)
    lower = np.zeros_like(success_rate)
    upper = np.zeros_like(success_rate)
    if hotfix_trial_success_index:
        # TODO(ahundt) currently the trial success values are inserted too early in the array. Fix then set hotfix param above to false
        trial_successes = np.insert(trial_successes, [0]*3, 0)
    for i in range(length - 1):
        start = max(i - window, 0)
        # get the number of trials that have passed starting with 0 at
        # the beginning of the trial window, by subtracting the
        # min trial count in the window from the current
        trial_window = trials[start:i+1] - np.min(trials[start:i+1])
        # get the number of successful trials that have passed starting with 0 at
        # the beginning of the trial window, by subtracting the
        # min successful trial count in the window from the current
        success_window = trial_successes[start:i+1] - np.min(trial_successes[start:i+1])
        # if success_window.shape[0] < window:
        #     print(window - success_window.shape[0])
        #     success_window = np.concatenate([success_window, np.zeros(window - success_window.shape[0])], axis=0)
        success_window_max = np.max(success_window)
        trial_window_max = np.max(trial_window)
        if trials.shape[0] >= window and i < window:
            trial_window_max = max(trial_window_max, np.max(trials[:window]))
        success_rate[i] = np.divide(success_window_max, trial_window_max, out=np.zeros(1), where=trial_window_max!=0.0)

    # TODO(ahundt) fix the discontinuities in the log from writing the success count at a slightly different time, remove median filter workaround
    if np.any(success_rate > 1.0):
        print('WARNING: BUG DETECTED, applying median filter to compensate for trial success time step offsets. '
              'The max is ' + str(np.max(success_rate)) + ' at index ' + str(np.argmax(success_rate)) +
              ' but the largest valid value is 1.0. You should look at the raw log data, '
              'fix the bug in the original code, and preprocess the raw data to correct this error.')
        # success_rate = np.clip(success_rate, 0, 1)
    success_rate = scipy.ndimage.median_filter(success_rate, 7)
    for i in range(length - 1):
        var = np.sqrt(success_rate[i] * (1 - success_rate[i]) / success_window.shape[0])
        lower[i] = success_rate[i] + 3*var
        upper[i] = success_rate[i] - 3*var
    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)
    # Print the best success rate ever, excluding actions before the initial window
    best_dict = best_success_rate(success_rate, window, 'trial success rate')
    return success_rate, lower, upper, best_dict


def get_grasp_success_rate(actions, rewards=None, window=200, reward_threshold=0.5):
    """Evaluate moving window of grasp success rate
    actions: Nx4 array of actions giving [id, rotation, i, j]
    rewards: an array of size N with the rewards associated with each action, only viable in pushing/grasping scenario,
             do not specify if placing is available, because a place action indicates the previous grasp was successful.
    """
    grasps = actions[:, 0] == utils.ACTION_TO_ID['grasp']
    if rewards is None:
        places = actions[:, 0] == utils.ACTION_TO_ID['place']
    length = np.min([rewards.shape[0], actions.shape[0]])
    success_rate = np.zeros(length - 1)
    lower = np.zeros_like(success_rate)
    upper = np.zeros_like(success_rate)
    for i in range(length - 1):
        start = max(i - window, 0)
        if rewards is None:
            # Where a place entry is True, the grasp on the previous action was successful
            successes = places[start+1: i+2][grasps[start:i+1]]
        else:
            successes = (rewards[start: i+1] > reward_threshold)[grasps[start:i+1]]
        grasp_count = grasps[start:i+1].sum()
        if successes.shape[0] < window and length > window and i < window:
            # Inital actions are zero filled, assuming an "infinite past of failure" before the first action.
            # print('extra zeros: ' + str(np.sum(grasps[i:window])))
            grasp_count =  grasps[start:min(start+window, grasps.shape[0])].sum()
        success_rate[i] = float(successes.sum()) / float(grasp_count) if grasp_count > 0 else 0.0
        var = np.sqrt(success_rate[i] * (1 - success_rate[i]) / successes.shape[0])
        lower[i] = success_rate[i] + 3*var
        upper[i] = success_rate[i] - 3*var
    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)
    # Print the best success rate ever, excluding actions before the initial window
    best_dict = best_success_rate(success_rate, window, 'grasp success rate')
    return success_rate, lower, upper, best_dict


def get_place_success_rate(stack_height, actions, include_push=False, window=200, hot_fix=False, max_height=4):
    """
    stack_heights: length N array of integer stack heights
    actions: Nx4 array of actions giving [id, rotation, i, j]
    hot_fix: fix the stack_height bug, where the trial didn't end on successful pushes, which reached a stack of 4.

    where id=0 is a push, id=1 is grasp, and id=2 is place.

    """
    if hot_fix:
        indices = np.logical_or(stack_height < 4, np.array([True] + list(stack_height[:-1] < 4)))
        actions = actions[:stack_height.shape[0]][indices]
        stack_height = stack_height[indices]

    if include_push:
        success_possible = actions[:, 0] == 2
    else:
        success_possible = np.logical_or(actions[:, 0] == 0, actions[:, 0] == 2)

    stack_height_increased = np.zeros_like(stack_height, np.bool)
    stack_height_increased[0] = False
    # the stack height increased if the next stack height is higher than the previous
    stack_height_increased[1:] = stack_height[1:] > stack_height[:-1]

    success_rate = np.zeros_like(stack_height)
    lower = np.zeros_like(success_rate)
    upper = np.zeros_like(success_rate)
    for i in range(stack_height.shape[0]):
        start = max(i - window, 0)
        successes = stack_height_increased[start:i+1][success_possible[start:i+1]]
        if stack_height.shape[0] > window and i < window:
            successes = np.concatenate([successes, np.zeros(window - i)], axis=0)
        success_rate[i] = successes.mean()
        success_rate[np.isnan(success_rate)] = 0
        var = np.sqrt(success_rate[i] * (1 - success_rate[i]) / successes.shape[0])
        lower[i] = success_rate[i] + 3*var
        upper[i] = success_rate[i] - 3*var
    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)
    # Print the best success rate ever, excluding actions before the initial window
    best_dict = best_success_rate(success_rate, window, 'place success rate')
    return success_rate, lower, upper, best_dict


def get_action_efficiency(stack_height, window=200, ideal_actions_per_trial=6, max_height=4):
    """Calculate the running action efficiency from successful trials.

    trials: array giving the number of trials up to iteration i (TODO: unused?)
    min_actions: ideal number of actions per trial

    Formula: successful_trial_count * ideal_actions_per_trial / window_size
    """
    # a stack is considered successful when the height is >= 4 blocks tall (~20cm)
    # success = np.rint(stack_height) == max_height
    # TODO(ahundt) it may be better to drop this function and modify get_trial_success_rate() to calculate: max(trial_successes)-min(trial_successes)/(window/ideal_actions_per_trial)
    success = stack_height >= max_height
    efficiency = np.zeros_like(stack_height, np.float64)
    lower = np.zeros_like(efficiency)
    upper = np.zeros_like(efficiency)
    for i in range(1, efficiency.shape[0]):
        start = max(i - window, 1)
        # window_size = min(i, window)
        window_size = np.array(min(i+1, window), np.float64)
        num_trials = success[start:i+1].sum()
        efficiency[i] = num_trials * ideal_actions_per_trial / window
        var = efficiency[i] / np.sqrt(window_size)
        lower[i] = efficiency[i] + 3*var
        upper[i] = efficiency[i] - 3*var
    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)
    # Print the best success rate ever, excluding actions before the initial window
    best_dict = best_success_rate(efficiency, window, 'action efficiency')
    return efficiency, lower, upper, best_dict


def get_grasp_action_efficiency(actions, rewards, reward_threshold=0.5, window=200, ideal_actions_per_trial=3):
    """Get grasp efficiency from when the trial count increases.

    """
    grasps = actions[:, 0] == 1
    length = np.min([rewards.shape[0], actions.shape[0]])
    efficiency = np.zeros(length, np.float64)
    lower = np.zeros_like(efficiency)
    upper = np.zeros_like(efficiency)
    for i in range(1, length):
        start = max(i - window, 0)
        window_size = np.array(min(i+1, window), np.float64)
        successful = rewards[start: i+1] > reward_threshold

        successful_grasps = np.array(successful[grasps[start:start+successful.shape[0]]].sum(), np.float64)
        # print(successfu)

        # print(successful_grasps)
        efficiency[i] = successful_grasps / window
        var = efficiency[i] / np.sqrt(window_size)
        lower[i] = efficiency[i] + 3*var
        upper[i] = efficiency[i] - 3*var
    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)
    # Print the best success rate ever, excluding actions before the initial window
    best_dict = best_success_rate(efficiency, window, 'grasp action efficiency')
    return efficiency, lower, upper, best_dict


def real_robot_speckle_noise_hotfix(heights, trial, trial_success, clearance, over_height_threshold=6.0):
    # length = min([heights.shape[0], trial.shape[0], trial_success.shape[0]])
    actions_with_height_noise = heights > over_height_threshold
    new_clearance = []
    for trial_it in clearance:
        recent_actions = actions_with_height_noise[int(trial_it) - 3:int(trial_it)]
        if not np.any(recent_actions):
            new_clearance += [trial_it]
    trial = np.array(utils.clearance_log_to_trial_count(new_clearance)).astype(np.int)
    heights[actions_with_height_noise] = 1.0
    return heights, trial, trial_success, clearance


def plot_it(log_dir, title, window=1000, colors=None,
            alpha=0.16, mult=100, max_iter=None, place=None, rasterized=True, clear_figure=True,
            apply_real_robot_speckle_noise_hotfix=False, num_preset_arrangements=None,
            label=None, categories=None, ylabel=None, save=True):
    if categories is None:
        categories = ['place_success', 'grasp_success', 'action_efficiency', 'trial_success']
    if colors is None:
        colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
    best_dict = {}
    stack_height_file = os.path.join(log_dir, 'transitions', 'stack-height.log.txt')
    if os.path.isfile(stack_height_file):
        heights = np.loadtxt(stack_height_file)
        rewards = None
        if place is None:
            place = True
    else:
        rewards = np.loadtxt(os.path.join(log_dir, 'transitions', 'reward-value.log.txt'))
        if place is None:
            place = False
    actions = np.loadtxt(os.path.join(log_dir, 'transitions', 'executed-action.log.txt'))
    trial_complete_indices = np.loadtxt(os.path.join(log_dir, 'transitions', 'clearance.log.txt'))
    print('trial_complete_indices: ' + str(trial_complete_indices))
    trials = np.array(utils.clearance_log_to_trial_count(trial_complete_indices)).astype(np.int)
    if window is None:
        # if window isn't defined, make it just shy of the full data length,
        # since log updates are delayed by a couple actions in some cases
        window = len(actions) - 4

    if max_iter is not None:
        if place:
            heights = heights[:max_iter]
        else:
            rewards = rewards[:max_iter]
        actions = actions[:max_iter]
        trials = trials[:max_iter]

    grasp_success_file = os.path.join(log_dir, 'transitions', 'grasp-success.log.txt')
    if os.path.isfile(grasp_success_file):
        grasp_rewards = np.loadtxt(grasp_success_file)
    else:
        # old versions of logged code don't have the grasp-success.log.txt file, data must be extracted from rewards.
        grasp_rewards = rewards

    # create and clear the figure
    if clear_figure:
        fig = plt.figure()
        fig.clf()
    else:
        # get the currently active figure
        fig = plt.gcf()
    # Plot the rate and variance of trial successes
    trial_success_file = os.path.join(log_dir, 'transitions', 'trial-success.log.txt')
    if os.path.isfile(trial_success_file):
        trial_successes = np.loadtxt(trial_success_file)
        if max_iter is not None:
            trial_successes = trial_successes[:max_iter]
        if apply_real_robot_speckle_noise_hotfix:
            clearance = np.loadtxt(os.path.join(log_dir, 'transitions', 'clearance.log.txt'))
            heights, trials, trial_successes, clearance = real_robot_speckle_noise_hotfix(heights, trials, trial_successes, clearance)
        if trial_successes.size > 0:
            trial_success_rate, trial_success_lower, trial_success_upper, best = get_trial_success_rate(trials, trial_successes, window=window)
            best_dict.update(best)
        if num_preset_arrangements is not None:
            best = count_preset_arrangements(trial_complete_indices, trial_successes, num_preset_arrangements)
            best_dict.update(best)
    # trial_reward_file = os.path.join(log_dir, 'transitions', 'trial-reward-value.log.txt')
    # if os.path.isfile(trial_reward_file):
    #     grasp_rewards = np.loadtxt(trial_reward_file)

    grasp_rate, grasp_lower, grasp_upper, best = get_grasp_success_rate(actions, rewards=grasp_rewards, window=window)
    best_dict.update(best)
    if place:
        if 'row' in log_dir or 'row' in title.lower():
            place_rate, place_lower, place_upper, best = get_place_success_rate(heights, actions, include_push=True, hot_fix=True, window=window)
        else:
            place_rate, place_lower, place_upper, best = get_place_success_rate(heights, actions, window=window)
        best_dict.update(best)
        eff, eff_lower, eff_upper, best = get_action_efficiency(heights, window=window)
        best_dict.update(best)
    else:
        eff, eff_lower, eff_upper, best = get_grasp_action_efficiency(actions, grasp_rewards, window=window)
        best_dict.update(best)

    if 'action_efficiency' in categories:
        plt.plot(mult*eff, color=colors[2], label=label or 'Action Efficiency')
        plt.fill_between(np.arange(1, eff.shape[0]+1),
                         mult*eff_lower, mult*eff_upper,
                         color=colors[2], alpha=alpha)
    if 'grasp_success' in categories:
        plt.plot(mult*grasp_rate, color=colors[0], label=label or 'Grasp Success Rate')
        # plt.fill_between(np.arange(1, grasp_rate.shape[0]+1),
        #                  mult*grasp_lower, mult*grasp_upper,
        #                  color=colors[0], alpha=alpha)
    if place and 'place_success' in categories:
        plt.plot(mult*place_rate, color=colors[1], label=label or 'Place Success Rate')
        # plt.fill_between(np.arange(1, place_rate.shape[0]+1),
                        #  mult*place_lower, mult*place_upper,
                        #  color=colors[1], alpha=alpha)

    if 'trial_success' in categories and os.path.isfile(trial_success_file) and trial_successes.size > 0:
        plt.plot(mult*trial_success_rate, color=colors[3], label=label or 'Trial Success Rate')
        # plt.fill_between(np.arange(1, trial_success_rate.shape[0]+1),
        #                  mult*trial_success_lower, mult*trial_success_upper,
        #                  color=colors[3], alpha=alpha)

    ax = plt.gca()
    plt.xlabel('Number of Actions')
    plt.ylabel('Mean % Over ' + str(window) + ' Actions, Higher is Better' if ylabel is None else ylabel)
    plt.title(title)
    plt.legend(loc='upper left')
    ax.yaxis.set_major_formatter(PercentFormatter())
    # we save the best stats and the generated plots in multiple locations for user convenience and backwards compatibility
    file_format = '.png'
    save_file = os.path.basename(log_dir + '-' + title).replace(':', '-').replace('.', '-').replace(',', '').replace(' ', '-') + '_success_plot'
    if save:
        print('saving plot: ' + save_file + file_format)
        plt.savefig(save_file + file_format, dpi=300, optimize=True)
        log_dir_fig_file = os.path.join(log_dir, save_file)
        plt.savefig(log_dir_fig_file + file_format, dpi=300, optimize=True)
        # plt.savefig(save_file + '.pdf')
        # this is a backwards compatibility location for best_stats.json
        best_stats_file = os.path.join(log_dir, 'data', 'best_stats.json')
        print('saving best stats to: ' + best_stats_file)
        with open(best_stats_file, 'w') as f:
            json.dump(best_dict, f, cls=utils.NumpyEncoder, sort_keys=True)
        # this is the more useful location for best_stats.json
        best_stats_file = os.path.join(log_dir, 'best_stats.json')
        print('saving best stats to: ' + best_stats_file)
        with open(best_stats_file, 'w') as f:
            json.dump(best_dict, f, cls=utils.NumpyEncoder, sort_keys=True)
    if clear_figure:
        plt.close(fig)
    return best_dict


def plot_compare(dirs, title, colors=None, labels=None, category='trial_success', **kwargs):
    if labels is None:
        labels = dirs
    kwargs['categories'] = [category]
    best_dicts = {}
    if colors is None:
        cmap = plt.get_cmap('viridis')
        colors = [[cmap(i / len(dirs))] * 4 for i, run_dir in enumerate(dirs)]
    for i, run_dir in enumerate(dirs):
        kwargs['clear_figure'] = i == 0
        kwargs['label'] = labels[i]
        kwargs['colors'] = colors[i]
        kwargs['save'] = i == len(dirs)-1
        best_dicts[run_dir] = plot_it(run_dir, title, **kwargs)
    return best_dicts # for some reason


if __name__ == '__main__':
    # window = 1000
    max_iter = None
    window = 1000
    plot_it('/home/costar/src/real_good_robot/logs/2020-02-24-01-16-21_Real-Push-and-Grasp-SPOT-Trial-Reward-Common-Sense-Testing', 'Sim to Real Pushing And Grasping, SPOT-Q',max_iter=None, window=None)

    # plot_it('/home/costar/src/real_good_robot/logs/2020-02-22-19-54-28_Real-Push-and-Grasp-SPOT-Trial-Reward-Common-Sense-Testing', 'Sim to Real Pushing And Grasping, SPOT-Q',max_iter=None, window=None)
    # plot_it('/home/costar/src/real_good_robot/logs/2020-02-23-11-43-55_Real-Push-and-Grasp-SPOT-Trial-Reward-Common-Sense-Training/2020-02-23-18-51-58_Real-Push-and-Grasp-SPOT-Trial-Reward-Common-Sense-Testing','Real Push and Grasp, SPOT-Q, Training', max_iter=None,window=None)
    # plot_it('/home/costar/src/real_good_robot/logs/2020-02-23-11-43-55_Real-Push-and-Grasp-SPOT-Trial-Reward-Common-Sense-Training', 'Real Push and Grasp, SPOT-Q, Training', max_iter=1000, window=500)
    ##############################################################
    #### IMPORTANT PLOT IN FINAL PAPER, data on costar workstation
    # best_dict = plot_compare(['./logs/2020-02-03-16-57-28_Sim-Stack-Trial-Reward-Common-Sense-Training',
    #                           './logs/2020-02-20-16-20-23_Sim-Stack-SPOT-Trial-Reward-Common-Sense-Training',
    #                           './logs/2020-02-03-16-58-06_Sim-Stack-Trial-Reward-Training'],
    #                           title='Effect of Action Space on Early Training Progress',
    #                           labels=['Dynamic with SPOT-Q',
    #                                   'Dynamic no SPOT-Q',
    #                                   'Standard'],
    #                           max_iter=3000, window=window,
    #                           ylabel='Mean Trial Success Rate Over ' + str(window) + ' Actions\nHigher is Better')

    ##############################################################
    # window = 200
    # best_dict = plot_compare(['./logs/2020-02-16-push-and-grasp-comparison/2020-02-16-21-33-59_Sim-Push-and-Grasp-SPOT-Trial-Reward-Common-Sense-Training',
    #                           './logs/2020-02-16-push-and-grasp-comparison/2020-02-16-21-37-47_Sim-Push-and-Grasp-SPOT-Trial-Reward-Training',
    #                           './logs/2020-02-16-push-and-grasp-comparison/2020-02-16-21-33-55_Sim-Push-and-Grasp-Two-Step-Reward-Training'],
    #                           title='Early Grasping Success Rate in Training', labels=['SPOT-Q (Dynamic Action Space)', 'SPOT (Standard Action Space)', 'VPG (Prior Work)'],
    #                           max_iter=2000, window=window,
    #                           category='grasp_success',
    #                           ylabel='Mean Grasp Success Rate Over ' + str(window) + ' Actions\nHigher is Better')
    # # best_dict = plot_it('./logs/2020-02-03-16-58-06_Sim-Stack-Trial-Reward-Training','Sim Stack, SPOT Trial Reward, Standard Action Space', window=1000)
    # best_dict = plot_it('./logs/2020-02-19-15-33-05_Real-Push-and-Grasp-SPOT-Trial-Reward-Common-Sense-Training', 'Real Push and Grasp, SPOT Reward, Common Sense', window=150)
    # best_dict = plot_it('./logs/2020-02-18-18-58-15_Real-Push-and-Grasp-Two-Step-Reward-Training', 'Real Push and Grasp, VPG', window=150)
    # best_dict = plot_it('./logs/2020-02-14-15-24-00_Sim-Rows-SPOT-Trial-Reward-Common-Sense-Testing', 'Sim Rows, SPOT Trial Reward, Common Sense, Testing', window=None)
    # Sim stats for final paper:
    # best_dict = plot_it('./logs/2020-02-11-15-53-12_Sim-Push-and-Grasp-Two-Step-Reward-Testing', 'Sim Push & Grasp, VPG, Challenging Arrangements', window=None, num_preset_arrangements=11)
    # best_dict = plot_it('./logs/2020-02-12-21-10-24_Sim-Rows-SPOT-Trial-Reward-Common-Sense-Testing', 'Sim Rows, SPOT Trial Reward, Common Sense, Testing', window=563)
    # print(best_dict)
    # log_dir = './logs/2020-01-20-11-40-56_Sim-Push-and-Grasp-Trial-Reward-Training'
    # log_dir = './logs/2020-01-20-14-25-13_Sim-Push-and-Grasp-Trial-Reward-Training'
    # log_dir = './logs/2020-02-03-14-47-16_Sim-Stack-Trial-Reward-Common-Sense-Training'
    # plot_it('./logs/2020-02-10-14-57-07_Real-Stack-SPOT-Trial-Reward-Common-Sense-Training','Real Stack, SPOT Reward, Common Sense, Training', window=200, max_iter=1000)
    # #############################################################
    # # REAL ROBOT STACKING run
    # plot_it('./logs/2020-02-09-11-02-57_Real-Stack-SPOT-Trial-Reward-Common-Sense-Training','Real Stack, SPOT-Q Dynamic Action Space, Training', window=500, max_iter=2500, apply_real_robot_speckle_noise_hotfix=True)
    # # Max trial success rate: 0.5833333333333334, at action iteration: 449. (total of 737 actions, max excludes first 200 actions)
    # # Max grasp success rate: 0.794392523364486, at action iteration: 289. (total of 750 actions, max excludes first 200 actions)
    # # Max place success rate: 0.7582417582417582, at action iteration: 119. (total of 751 actions, max excludes first 200 actions)
    # # Max action efficiency: 0.3, at action iteration: 37. (total of 751 actions, max excludes first 200 actions)
    # #############################################################
    # # Here is the good & clean simulation common sense push & grasp densenet plot with SPOT reward, run on the costar workstation.
    # # It can basically complete trials 100% of the time within 400 actions!
    # plot_it('./logs/2020-02-07-14-43-44_Sim-Push-and-Grasp-Trial-Reward-Common-Sense-Training','Sim Push and Grasp, SPOT Reward, Common Sense, Training', window=200, max_iter=2500)
    # # plot_it(log_dir, log_dir, window=window, max_iter=max_iter)
    # #############################################################
    # # ABSOLUTE BEST STACKING RUN AS OF 2020-02-04, on costar workstation
    # log_dir = './logs/2020-02-03-16-57-28_Sim-Stack-Trial-Reward-Common-Sense-Training'
    # # plot_it(log_dir, 'Sim Stack, Trial Reward, Common Sense, Training', window=window, max_iter=max_iter)
    # plot_it(log_dir,'Sim Stack, SPOT Reward, Common Sense, Training', window=window, max_iter=4000)
    # #############################################################

    # log_dir = './logs/2020-01-22-19-10-50_Sim-Push-and-Grasp-Two-Step-Reward-Training'
    # log_dir = './logs/2020-01-22-22-50-00_Sim-Push-and-Grasp-Two-Step-Reward-Training'
    # log_dir = './logs/2020-02-03-17-35-43_Sim-Push-and-Grasp-Two-Step-Reward-Training'
    # log_dir = './logs/2020-02-06-14-41-48_Sim-Stack-Trial-Reward-Common-Sense-Training'
    # plot_it(log_dir, log_dir, window=window, max_iter=max_iter)

    # # log_dir = './logs/2019-12-31-20-17-06'
    # # log_dir = './logs/2020-01-01-14-55-17'
    # log_dir = './logs/2020-01-08-17-03-58'
    # log_dir = './logs/2020-01-08-17-03-58-test-resume'
    # # Stacking 0.
    # log_dir = './logs/2020-01-12-12-33-41'
    # # Creating data logging session: /home/costar/src/real_good_robot/logs/2020-01-12-12-33-41 # this run had a problem

    # # ± /usr/bin/python3 /home/costar/src/real_good_robot/main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 8 --push_rewards --experience_replay --explore_rate_decay --trial_reward --save_visualizations --skip_noncontact_actions --check_z_height --tcp_port 19997 --place --future_reward_discount 0.65
    # # Creating data logging session: /home/costar/src/real_good_robot/logs/2020-01-12-17-56-46
    # # log_dir = './logs/2020-01-13-10-15-49' # this run stopped after 1750 actions
    # # Creating data logging session: /home/costar/src/real_good_robot/logs/2020-01-13-10-15-49 # stopped after 1750 actions
    # log_dir = './logs/2020-01-14-18-36-16'
    # # Creating data logging session: /home/costar/src/real_good_robot/logs/2020-01-14-18-36-16
    # log_dir = './logs/2020-01-15-15-44-39'

    # title = 'Stack 4 Blocks, Trial Reward 0.65, Training'
    # # plot_it(log_dir, title, window=window, max_iter=max_iter, place=True)
    # plot_it(log_dir, title, window=window, max_iter=max_iter, place=True)
    # # this is a solid but slow training trial_reward grasp and push run without symmetry
    # # title = 'Push and Grasp, Trial Reward, No Symmetry, Training'
    # # log_dir = './logs/2020-01-06-19-15-55'
    # # plot_it(log_dir, title, window=window, max_iter=max_iter, place=False)
    # run = 2
    # if run == 0:
    #     title = 'Rows, Trial Reward 0.5, No Symmetry, Training'
    #     # log_dir = './logs/2020-01-07-17-53-42' # some progress, not complete
    #     # log_dir = './logs/2020-01-08-17-08-57' # run killed early
    #     log_dir = './logs/2020-01-09-12-54-53'
    #     # Training iteration: 22769
    #     # Current count of pixels with stuff: 2513.0 threshold below which the scene is considered empty: 1200
    #     # WARNING variable mismatch num_trials + 1: 3118 nonlocal_variables[stack].trial: 3359
    #     # Change detected: True (value: 2799)
    #     # Primitive confidence scores: 4.359684 (push), 2.701111 (grasp), 4.351819 (place)
    #     # Strategy: exploit (exploration probability: 0.100000)
    #     # Action: push at (1, 99, 10)
    #     # Real Robot push at (-0.704000, -0.026000, 0.000994) angle: 0.392699
    #     # Trainer.get_label_value(): Current reward: 0.750000 Current reward multiplier: 1.000000 Predicted Future reward: 4.402410 Expected reward: 0.750000 + 0.500000 x 4.402410 = 2.951205
    #     # Trial logging complete: 3117 --------------------------------------------------------------
    #     # Training loss: 0.897331
    #     # /home/ahundt/src/real_good_robot/logs/2020-01-08-18-16-12
    #     plot_it(log_dir, title, window=window, max_iter=max_iter, place=True)

    # if run == 1:
    #     title = 'Rows, Trial Reward 0.65, No Symmetry, Training'
    #     # ± export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir 'objects/blocks' --num_obj 4  --push_rewards --experience_replay --explore_rate_decay --trial_reward --tcp_port 19997 --place --check_row --future_reward_discount 0.65
    #     # Creating data logging session: /home/ahundt/src/real_good_robot/logs/2020-01-11-19-54-58
    #     log_dir = './logs/2020-01-11-19-54-58'
    #     # /home/ahundt/src/real_good_robot/logs/2020-01-08-18-16-12
    #     plot_it(log_dir, title, window=window, max_iter=max_iter, place=True)

    # if run == 2:
    #     title = 'Rows, Trial Reward 0.65, No Symmetry, Training'
    #     # ± export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir 'objects/blocks' --num_obj 4  --push_rewards --experience_replay --explore_rate_decay --trial_reward --tcp_port 19997 --place --check_row --future_reward_discount 0.65
    #     # Creating data logging session: /home/ahundt/src/real_good_robot/logs/2020-01-12-17-42-46
    #     # Creating data logging session: /home/ahundt/src/real_good_robot/logs/2020-01-12-17-45-22
    #     log_dir = './logs/2020-01-12-17-45-22'
    #     plot_it(log_dir, title, window=window, max_iter=max_iter, place=True)
