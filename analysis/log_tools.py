import csv
import pandas 
import pathlib
import numpy as np
import pdb

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
    clearance_path = log_dir.joinpath("clearance.log.csv")
    trial_num_path = log_dir.joinpath("trial.log.csv") 
    trial_success_path = log_dir.joinpath("trial-success.log.csv")
    clearance_data = read_csv(clearance_path)
    trial_num_data = read_csv(trial_num_path)
    trial_success_data = read_csv(trial_success_path)    


    trial_lens = []
    successful_trial_lens = []
    trial_start = 0
    for i in range(len(clearance_data)-1):
        trial_end = int(float(clearance_data[i][0]))
        trial_length = trial_end - trial_start 

        print(trial_end)
        print(len(trial_success_data))
        curr_num_successes = int(float(trial_success_data[trial_end-2][0]))
        next_num_successes = int(float(trial_success_data[trial_end-1][0]))

        was_success = next_num_successes > curr_num_successes
        if was_success: 
            successful_trial_lens.append(trial_length)
        trial_lens.append(trial_length)
        trial_start = trial_end
    
    print(successful_trial_lens)
    return np.mean(successful_trial_lens), np.mean(trial_lens)



if __name__ == "__main__":

    path = pathlib.Path("/home/elias/src/real_good_robot/logs/row_munge_test")
    #perc = get_correct_color_percentage(path)
    #print(f"path {path} has perc {perc}")

    success_num_actions, total_num_actions = get_number_actions_to_complete(path)

    print(f"path {path} has {success_num_actions} to complete successful trial ")

