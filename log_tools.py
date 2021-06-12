import csv
import pandas 
import pathlib
import numpy as np
import pdb
import argparse

## metrics to get
## 2. number of actions to complete a stack/row 
## 3. error percentages (topple, incorrect ordering, time out)
    # Can i get these right now
    # topple: trial.log.txt + stack_height.log.txt can detect when stack height decreases
    # incorrect ordering: trial.log.txt + stack_height.log.txt + grasp_color can give when wrong color picked up and stacked on top
    # timed out: trial.log.txt and trial-success.log.txt can give failures where max limit reached 

def read_csv(path):
    with open(path) as f1:
        data = f1.readlines()[1:]
    data = [line.split(", ") for line in data]
    return data 

def read_txt(path):
    with open(path) as f1:
        data = f1.readlines()
    return data

def get_correct_color_percentage(log_dir):
    ## 1. percentage of correct color pickups
    log_dir = log_dir.joinpath("transitions")
    color_path = log_dir.joinpath("grasp-color-success.log.csv")
    action_path = log_dir.joinpath("executed-action.log.csv")
    grasp_path = log_dir.joinpath("grasp-success.log.csv")

    color_data = read_csv(color_path)
    action_data = read_csv(action_path)
    grasp_data = read_csv(grasp_path)
    assert(len(color_data) == len(action_data) == len(grasp_data))
    total = 0
    n_successful = 0
    for color_success, grasp_success, action_type in zip(color_data, grasp_data, action_data):
        action_taken = action_type[0]
        color_success = color_success[0]
        grasp_success = grasp_success[0]
        if int(float(action_taken)) == 1 and int(float(grasp_success)) == 1:
            if int(float(color_success)) == 1:
                n_successful += 1
            total += 1
    return n_successful/total

def get_number_actions_to_complete(log_dir):
    log_dir = log_dir.joinpath("transitions")
    #clearance_path = log_dir.joinpath("clearance.log.csv")
    clearance_path = log_dir.joinpath("clearance.log.txt")
    trial_num_path = log_dir.joinpath("trial.log.csv") 
    trial_success_path = log_dir.joinpath("trial-success.log.csv")
    #clearance_data = read_csv(clearance_path)
    clearance_data = read_txt(clearance_path)
    trial_num_data = read_csv(trial_num_path)
    trial_success_data = read_csv(trial_success_path)    


    trial_lens = []
    successful_trial_lens = []
    trial_start = 0
    for i in range(len(clearance_data)-1):
        trial_end = int(float(clearance_data[i][0]))
        trial_length = trial_end - trial_start 

        curr_num_successes = int(float(trial_success_data[trial_end-2][0]))
        next_num_successes = int(float(trial_success_data[trial_end-1][0]))
        was_success = next_num_successes > curr_num_successes
        if was_success: 
            successful_trial_lens.append(trial_length)
        trial_lens.append(trial_length)
        trial_start = trial_end
    
    return np.mean(successful_trial_lens), np.mean(trial_lens), len(successful_trial_lens)

def error_analysis(log_dir):
    log_dir = log_dir.joinpath("transitions")
    #clearance_path = log_dir.joinpath("clearance.log.csv")
    clearance_path = log_dir.joinpath("clearance.log.txt")
    trial_num_path = log_dir.joinpath("trial.log.csv") 
    trial_success_path = log_dir.joinpath("trial-success.log.csv")
    stack_height_path = log_dir.joinpath("stack-height.log.csv")
    #clearance_data = read_csv(clearance_path)
    clearance_data = read_txt(clearance_path)
    trial_num_data = read_csv(trial_num_path)
    trial_success_data = read_csv(trial_success_path)    
    stack_height_data = read_csv(stack_height_path)

    # topple: look at stack height reductions
    n_toppled = 0
    n_incorrect_order = 0
    n_timeout = 0
    total_failures = 0
    trial_start = 0 
    for i in range(len(clearance_data)-1):
        trial_end = int(float(clearance_data[i][0]))
        trial_length = trial_end - trial_start 

        curr_num_successes = int(float(trial_success_data[trial_end-2][0]))
        next_num_successes = int(float(trial_success_data[trial_end-1][0]))
        was_success = next_num_successes > curr_num_successes
        if not was_success:
            total_failures += 1
            if trial_length > 29:
                n_timeout += 1
                continue 
            stack_height_before = int(float(stack_height_data[trial_end-2][0]))
            stack_height_after = int(float(stack_height_data[trial_end-1][0]))
            if stack_height_before > stack_height_after: 
                n_toppled += 1
                continue  

            elif stack_height_before <= stack_height_after:
                n_incorrect_order += 1
                continue 
        trial_start = trial_end - 1
    return n_timeout / total_failures, n_incorrect_order / total_failures, n_toppled / total_failures 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to log dir")
    parser.add_argument("--stacks", action="store_true", help="set to true if analyzing stack dir")
    args = parser.parse_args()

    path = pathlib.Path(args.path)

    perc_grasped = get_correct_color_percentage(path)
    success_num_actions, total_num_actions, total_num_successes = get_number_actions_to_complete(path)

    if args.stacks:
        perc_timeout, perc_incorrect, perc_toppled = error_analysis(path)


    print(f"Report for logdir: {args.path}")
    print(f"\tNumber of successes: {total_num_successes}")
    print(f"\tPercentage correct grasps: {100* perc_grasped:.2f}")
    print(f"\tAvg. # actions: {success_num_actions:.2f}")
    if args.stacks:
        print(f"\tError Analysis:")
        print(f"\t\tTimeout: {100*perc_timeout:.2f}")
        print(f"\t\tIncorrect Order: {100*perc_incorrect:.2f}")
        print(f"\t\tToppled: {100*perc_toppled:.2f}")


