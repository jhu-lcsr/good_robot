import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from glob import glob

def get_trial_success_rate(trials, trial_successes, window=200):
    """Evaluate moving window of grasp success rate
    trials: Nx1 array of the current total trial count at that action
    trial_successes: Nx1 array of the current total successful trial count at the time of that action

    """
    length = np.min([trials.shape[0], trial_successes.shape[0]])
    success_rate = np.zeros(length - 1)
    lower = np.zeros_like(success_rate)
    upper = np.zeros_like(success_rate)
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
        success_rate[i] = np.max(success_window) / np.max(trial_window)
        var = np.sqrt(success_rate[i] * (1 - success_rate[i]) / success_window.shape[0])
        lower[i] = success_rate[i] + 3*var
        upper[i] = success_rate[i] - 3*var
    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)
    return success_rate, lower, upper

def get_grasp_success_rate(actions, rewards=None, window=200, reward_threshold=0.5):
    """Evaluate moving window of grasp success rate
    actions: Nx4 array of actions giving [id, rotation, i, j]

    """
    grasps = actions[:, 0] == 1
    if rewards is None:
        places = actions[:, 0] == 2
    length = np.min([rewards.shape[0], actions.shape[0]])
    success_rate = np.zeros(length - 1)
    lower = np.zeros_like(success_rate)
    upper = np.zeros_like(success_rate)
    for i in range(length - 1):
        start = max(i - window, 0)
        if rewards is None:
            successes = places[start+1: i+2][grasps[start:i+1]]
        else:
            successes = (rewards[start: i+1] > reward_threshold)[grasps[start:i+1]]
        success_rate[i] = successes.mean()
        var = np.sqrt(success_rate[i] * (1 - success_rate[i]) / successes.shape[0])
        lower[i] = success_rate[i] + 3*var
        upper[i] = success_rate[i] - 3*var
    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)
    return success_rate, lower, upper

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
    stack_height_increased[1:] = stack_height[1:] > stack_height[:-1]

    success_rate = np.zeros_like(stack_height)
    lower = np.zeros_like(success_rate)
    upper = np.zeros_like(success_rate)
    for i in range(stack_height.shape[0]):
        start = max(i - window, 0)
        successes = stack_height_increased[start:i+1][success_possible[start:i+1]]
        success_rate[i] = successes.mean()
        success_rate[np.isnan(success_rate)] = 0
        var = np.sqrt(success_rate[i] * (1 - success_rate[i]) / successes.shape[0])
        lower[i] = success_rate[i] + 3*var
        upper[i] = success_rate[i] - 3*var
    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)
    return success_rate, lower, upper

def get_action_efficiency(stack_height, window=200, ideal_actions_per_trial=6, max_height=4):
    """Calculate the running action efficiency from successful trials.

    trials: array giving the number of trials up to iteration i (TODO: unused?)
    min_actions: ideal number of actions per trial

    Formula: successful_trial_count * ideal_actions_per_trial / window_size
    """

    success = np.rint(stack_height) == max_height
    efficiency = np.zeros_like(stack_height, np.float64)
    lower = np.zeros_like(efficiency)
    upper = np.zeros_like(efficiency)
    for i in range(1, efficiency.shape[0]):
        start = max(i - window, 1)
        # window_size = min(i, window)
        window_size = np.array(min(i+1, window), np.float64)
        num_trials = success[start:i+1].sum()
        efficiency[i] = num_trials * ideal_actions_per_trial / window_size
        var = efficiency[i] / np.sqrt(window_size)
        lower[i] = efficiency[i] + 3*var
        upper[i] = efficiency[i] - 3*var
    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)
    return efficiency, lower, upper

def get_grasp_action_efficiency(actions, rewards, reward_threshold=0.5, window=200, ideal_actions_per_trial=3):
    """Get grasp efficiency from when the trial count increases.

    """
    grasps = actions[:, 0] == 1
    efficiency = np.zeros_like(rewards, np.float64)
    lower = np.zeros_like(efficiency)
    upper = np.zeros_like(efficiency)
    for i in range(efficiency.shape[0]):
        start = max(i - window, 0)
        window_size = np.array(min(i+1, window), np.float64)
        successful = rewards[start: i+1] > reward_threshold
        successful_grasps = np.array(successful[grasps[start:start+successful.shape[0]]].sum(), np.float64)
        # print(successful_grasps)
        efficiency[i] = successful_grasps / window_size
        var = efficiency[i] / np.sqrt(window_size)
        lower[i] = efficiency[i] + 3*var
        upper[i] = efficiency[i] - 3*var
    lower = np.clip(lower, 0, 1)
    upper = np.clip(upper, 0, 1)
    return efficiency, lower, upper

def plot_it(log_dir, title, window=1000, colors=['tab:blue', 'tab:green', 'tab:orange', 'tab:purple'], alpha=0.35, mult=100, max_iter=None, place=False, rasterized=True):
    if place:
        heights = np.loadtxt(os.path.join(log_dir, 'transitions', 'stack-height.log.txt'))
        rewards = None
    else:
        rewards = np.loadtxt(os.path.join(log_dir, 'transitions', 'reward-value.log.txt'))
    actions = np.loadtxt(os.path.join(log_dir, 'transitions', 'executed-action.log.txt'))
    trials = np.loadtxt(os.path.join(log_dir, 'transitions', 'trial.log.txt'))

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

    # trial_reward_file = os.path.join(log_dir, 'transitions', 'trial-reward-value.log.txt')
    # if os.path.isfile(trial_reward_file):
    #     grasp_rewards = np.loadtxt(trial_reward_file)

    grasp_rate, grasp_lower, grasp_upper = get_grasp_success_rate(actions, rewards=grasp_rewards, window=window)
    if place:
        if 'row' in log_dir or 'row' in title.lower():
            place_rate, place_lower, place_upper = get_place_success_rate(heights, actions, include_push=True, hot_fix=True, window=window)
        else:
            place_rate, place_lower, place_upper = get_place_success_rate(heights, actions, window=window)
        eff, eff_lower, eff_upper = get_action_efficiency(heights, window=window)
    else:
        eff, eff_lower, eff_upper = get_grasp_action_efficiency(actions, grasp_rewards, window=window)

    plt.plot(mult*grasp_rate, color=colors[0], label='Grasp Success Rate')
    if place:
        plt.plot(mult*place_rate, color=colors[1], label='Place Success Rate')
    plt.plot(mult*eff, color=colors[2], label='Action Efficiency')

    plt.fill_between(np.arange(1, grasp_rate.shape[0]+1),
                     mult*grasp_lower, mult*grasp_upper,
                     color=colors[0], alpha=alpha)
    if place:
        plt.fill_between(np.arange(1, place_rate.shape[0]+1),
                         mult*place_lower, mult*place_upper,
                         color=colors[1], alpha=alpha)
    plt.fill_between(np.arange(1, eff.shape[0]+1),
                     mult*eff_lower, mult*eff_upper,
                     color=colors[2], alpha=alpha)
    
    # Plot the rate and variance of trial successes
    trial_success_file = os.path.join(log_dir, 'transitions', 'trial-success.log.txt')
    if os.path.isfile(trial_success_file):
        trial_successes = np.loadtxt(trial_success_file)
        if max_iter is not None:
            trial_successes = trial_successes[:max_iter]
        trial_success_rate, trial_success_lower, trial_success_upper = get_trial_success_rate(trials, trial_successes, window=window)
        plt.plot(mult*trial_success_rate, color=colors[3], label='Trial Success Rate')
        plt.fill_between(np.arange(1, trial_success_rate.shape[0]+1),
                        mult*trial_success_lower, mult*trial_success_upper,
                        color=colors[0], alpha=alpha)

    ax = plt.gca()
    plt.xlabel('Number of Actions')
    plt.ylabel('Mean % Over ' + str(window) + ' Actions, Higher is Better')
    plt.title(title)
    plt.legend(loc='upper left')
    ax.yaxis.set_major_formatter(PercentFormatter())
    save_file = os.path.basename(log_dir + '-' + title).replace(':', '-').replace('.', '-').replace(',','').replace(' ','-') + '_success_plot.png'
    print('saving plot: ' + save_file)
    plt.savefig(save_file)
    


if __name__ == '__main__':
    window = 500
    max_iter = None
    # log_dir = './logs/2019-12-31-20-17-06'
    # log_dir = './logs/2020-01-01-14-55-17'
    log_dir = './logs/2020-01-08-17-03-58'
    log_dir = './logs/2020-01-08-17-03-58-test-resume'
    # Stacking 0.
    log_dir = './logs/2020-01-12-12-33-41'
    # Creating data logging session: /home/costar/src/real_good_robot/logs/2020-01-12-12-33-41 # this run had a problem

    # ± /usr/bin/python3 /home/costar/src/real_good_robot/main.py --is_sim --obj_mesh_dir objects/blocks --num_obj 8 --push_rewards --experience_replay --explore_rate_decay --trial_reward --save_visualizations --skip_noncontact_actions --check_z_height --tcp_port 19997 --place --future_reward_discount 0.65
    # Creating data logging session: /home/costar/src/real_good_robot/logs/2020-01-12-17-56-46
    # log_dir = './logs/2020-01-13-10-15-49' # this run stopped after 1750 actions
    # Creating data logging session: /home/costar/src/real_good_robot/logs/2020-01-13-10-15-49 # stopped after 1750 actions
    log_dir = './logs/2020-01-14-18-36-16'
    # Creating data logging session: /home/costar/src/real_good_robot/logs/2020-01-14-18-36-16
    log_dir = './logs/2020-01-15-15-44-39'

    title = 'Stack 4 Blocks, Trial Reward 0.65, Training'
    plot_it(log_dir, title, window=window, max_iter=max_iter, place=True)
    # this is a solid but slow training trial_reward grasp and push run without symmetry
    # title = 'Push and Grasp, Trial Reward, No Symmetry, Training'
    # log_dir = './logs/2020-01-06-19-15-55'
    # plot_it(log_dir, title, window=window, max_iter=max_iter, place=False)
    run = 2
    if run == 0:
        title = 'Rows, Trial Reward 0.5, No Symmetry, Training'
        # log_dir = './logs/2020-01-07-17-53-42' # some progress, not complete
        # log_dir = './logs/2020-01-08-17-08-57' # run killed early
        log_dir = './logs/2020-01-09-12-54-53'
        # Training iteration: 22769
        # Current count of pixels with stuff: 2513.0 threshold below which the scene is considered empty: 1200
        # WARNING variable mismatch num_trials + 1: 3118 nonlocal_variables[stack].trial: 3359
        # Change detected: True (value: 2799)
        # Primitive confidence scores: 4.359684 (push), 2.701111 (grasp), 4.351819 (place)
        # Strategy: exploit (exploration probability: 0.100000)
        # Action: push at (1, 99, 10)
        # Real Robot push at (-0.704000, -0.026000, 0.000994) angle: 0.392699
        # Trainer.get_label_value(): Current reward: 0.750000 Current reward multiplier: 1.000000 Predicted Future reward: 4.402410 Expected reward: 0.750000 + 0.500000 x 4.402410 = 2.951205
        # Trial logging complete: 3117 --------------------------------------------------------------
        # Training loss: 0.897331
        # /home/ahundt/src/real_good_robot/logs/2020-01-08-18-16-12
        plot_it(log_dir, title, window=window, max_iter=max_iter, place=True)

    if run == 1:
        title = 'Rows, Trial Reward 0.65, No Symmetry, Training'
        # ± export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir 'objects/blocks' --num_obj 4  --push_rewards --experience_replay --explore_rate_decay --trial_reward --tcp_port 19997 --place --check_row --future_reward_discount 0.65
        # Creating data logging session: /home/ahundt/src/real_good_robot/logs/2020-01-11-19-54-58
        log_dir = './logs/2020-01-11-19-54-58'
        # /home/ahundt/src/real_good_robot/logs/2020-01-08-18-16-12
        plot_it(log_dir, title, window=window, max_iter=max_iter, place=True)

    if run == 2:
        title = 'Rows, Trial Reward 0.65, No Symmetry, Training'
        # ± export CUDA_VISIBLE_DEVICES="0" && python3 main.py --is_sim --obj_mesh_dir 'objects/blocks' --num_obj 4  --push_rewards --experience_replay --explore_rate_decay --trial_reward --tcp_port 19997 --place --check_row --future_reward_discount 0.65
        # Creating data logging session: /home/ahundt/src/real_good_robot/logs/2020-01-12-17-42-46
        # Creating data logging session: /home/ahundt/src/real_good_robot/logs/2020-01-12-17-45-22
        log_dir = './logs/2020-01-12-17-45-22'
        plot_it(log_dir, title, window=window, max_iter=max_iter, place=True)
