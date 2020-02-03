#!/usr/bin/env python

import time
import os
import random
import threading
import argparse
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils
from utils import ACTION_TO_ID
from utils import ID_TO_ACTION
import plot
import json


def run_title(args):
    """
    # Returns

    title, dirname
    """
    title = ''
    title += 'Sim ' if args.is_sim else 'Real '
    if args.place:
        title += 'Stack, '
    if args.check_row:
        title += 'Rows, '
    if not args.place and not args.check_row:
        title += 'Push and Grasp, '
    if args.trial_reward:
        title += 'Trial Reward, '
    else:
        title += 'Two Step Reward, '
    if args.common_sense:
        title += 'Common Sense, '
    title += 'Testing' if args.is_testing else 'Training'

    save_file = os.path.basename(title).replace(':', '-').replace('.', '-').replace(',','').replace(' ','-')
    dirname = utils.timeStamped(save_file)
    return title, dirname

# killeen: this is defining the goal
class StackSequence(object):
    def __init__(self, num_obj, is_goal_conditioned_task=True, trial=0, total_steps=1):
        """ Oracle to choose a sequence of specific color objects to interact with.

        Generates one hot encodings for a list of objects of the specified length.
        Can be used for stacking or simply grasping specific objects.

        # Member Variables

        num_obj: the number of objects to manage. Each object is assumed to be in a list indexed from 0 to num_obj.
        is_goal_conditioned_task: do we care about which specific object we are using
        object_color_sequence: to get the full order of the current stack goal.

        """
        self.num_obj = num_obj
        self.is_goal_conditioned_task = is_goal_conditioned_task
        self.trial = trial
        self.reset_sequence()
        self.total_steps = total_steps

    def reset_sequence(self):
        """ Generate a new sequence of specific objects to interact with.
        """
        if self.is_goal_conditioned_task:
            # 3 is currently the red block
            # object_color_index = 3
            self.object_color_index = 0

            # Choose a random sequence to stack
            self.object_color_sequence = np.random.permutation(self.num_obj)
            # TODO(ahundt) This might eventually need to be the size of robot.stored_action_labels, but making it color-only for now.
            self.object_color_one_hot_encodings = []
            for color in self.object_color_sequence:
                object_color_one_hot_encoding = np.zeros((self.num_obj))
                object_color_one_hot_encoding[color] = 1.0
                self.object_color_one_hot_encodings.append(object_color_one_hot_encoding)
        else:
            self.object_color_index = None
            self.object_color_one_hot_encodings = None
            self.object_color_sequence = None
        self.trial += 1

    def current_one_hot(self):
        """ Return the one hot encoding for the current specific object.
        """
        return self.object_color_one_hot_encodings[self.object_color_index]

    def sequence_one_hot(self):
        """ Return the one hot encoding for the entire stack sequence.
        """
        return np.concatenate(self.object_color_one_hot_encodings)

    def current_sequence_progress(self):
        """ How much of the current stacking sequence we have completed.

        For example, if the sequence should be [0, 1, 3, 2].
        At initialization this will return [0].
        After one next() calls it will return [0, 1].
        After two next() calls it will return [0, 1, 3].
        After three next() calls it will return [0, 1, 3, 2].
        After four next() calls a new sequence will be generated and it will return one element again.
        """
        if self.is_goal_conditioned_task:
            return self.object_color_sequence[:self.object_color_index+1]
        else:
            return None

    def next(self):
        self.total_steps += 1
        if self.is_goal_conditioned_task:
            self.object_color_index += 1
            if not self.object_color_index < self.num_obj:
                self.reset_sequence()


def main(args):
    # TODO(ahundt) move main and process_actions() to a class?

    # --------------- Setup options ---------------
    is_sim = args.is_sim # Run in simulation?
    obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) if is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj = args.num_obj if is_sim else None # Number of objects to add to simulation
    num_extra_obj = args.num_extra_obj if is_sim else None
    if num_obj is not None:
        num_obj += num_extra_obj
    tcp_host_ip = args.tcp_host_ip if not is_sim else None # IP and port to robot arm as TCP client (UR5)
    tcp_port = args.tcp_port  # TODO(killeen) change the rest of these?
    rtc_host_ip = args.rtc_host_ip if not is_sim else None # IP and port to robot arm as real-time client (UR5)
    rtc_port = args.rtc_port if not is_sim else None
    if is_sim:
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.5]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    else:
        # Corner near window on robot base side
        # [0.47984089 0.34192974 0.02173636]
        # Corner on the side of the cameras and far from the window
        # [ 0.73409861 -0.45199446 -0.00229499]
        # Dimensions of workspace should be 448 mm x 448 mm. That's 224x224 pixels with each pixel being 2mm x2mm.
        workspace_limits = np.asarray([[0.376, 0.824], [-0.264, 0.184], [-0.07, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)

        # Original visual pushing graping paper workspace definition
        # workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    heightmap_resolution = args.heightmap_resolution # Meters per pixel of heightmap
    random_seed = args.random_seed
    force_cpu = args.force_cpu
    flops = args.flops
    show_heightmap = args.show_heightmap

    # ------------- Algorithm options -------------
    method = args.method # 'reactive' (supervised learning) or 'reinforcement' (reinforcement learning ie Q-learning)
    push_rewards = args.push_rewards if method == 'reinforcement' else None  # Use immediate rewards (from change detection) for pushing?
    future_reward_discount = args.future_reward_discount
    experience_replay_enabled = args.experience_replay # Use prioritized experience replay?
    trial_reward = args.trial_reward
    heuristic_bootstrap = args.heuristic_bootstrap # Use handcrafted grasping algorithm when grasping fails too many times in a row?
    explore_rate_decay = args.explore_rate_decay
    grasp_only = args.grasp_only
    check_row = args.check_row
    check_z_height = args.check_z_height
    check_z_height_goal = args.check_z_height_goal
    pretrained = not args.random_weights
    max_iter = args.max_iter
    no_height_reward = args.no_height_reward
    transfer_grasp_to_place = args.transfer_grasp_to_place
    neural_network_name = args.nn
    disable_situation_removal = args.disable_situation_removal
    evaluate_random_objects = args.evaluate_random_objects
    skip_noncontact_actions = args.skip_noncontact_actions
    common_sense = args.common_sense
    disable_two_step_backprop = args.disable_two_step_backprop

    # -------------- Test grasping options --------------
    is_testing = args.is_testing
    max_test_trials = args.max_test_trials # Maximum number of test runs per case/scenario
    test_preset_cases = args.test_preset_cases
    trials_per_case = 1
    show_preset_cases_then_exit = args.show_preset_cases_then_exit
    if show_preset_cases_then_exit:
        test_preset_cases = True
    if test_preset_cases:
        if args.test_preset_file:
            # load just one specific file
            preset_files = [os.path.abspath(args.test_preset_file)]
        else:
            # load a directory of files
            preset_files = os.listdir(args.test_preset_dir)
            preset_files = [os.path.abspath(os.path.join(args.test_preset_dir, filename)) for filename in preset_files]
            preset_files = sorted(preset_files)
        trials_per_case = max_test_trials
        # run each preset file max_test_trials times.
        max_test_trials *= len(preset_files)
        test_preset_file = preset_files[0]
    else:
        preset_files = None
        test_preset_file = None

    # ------ Pre-loading and logging options ------
    if args.resume == 'last':
        dirs = [os.path.join(os.path.abspath('logs'), p) for p in os.listdir(os.path.abspath('logs'))]
        dirs = list(filter(os.path.isdir, dirs))
        if dirs:
            continue_logging = True
            logging_directory = sorted(dirs)[-1]
        else:
            print('no logging dirs to resume, starting new run')
            continue_logging = False
            logging_directory = os.path.abspath('logs')
    elif args.resume:
        continue_logging = True
        logging_directory = os.path.abspath(args.resume)
    else:
        continue_logging = False
        logging_directory = os.path.abspath('logs')

    snapshot_file = os.path.abspath(args.snapshot_file) if args.snapshot_file else ''
    if continue_logging and not snapshot_file:
        snapshot_file = os.path.join(logging_directory, 'models', 'snapshot.reinforcement.pth')
        print('loading snapshot file: ' + snapshot_file)
        if not os.path.isfile(snapshot_file):
            snapshot_file = os.path.join(logging_directory, 'models', 'snapshot-backup.reinforcement.pth')
            print('snapshot file does not exist, trying backup: ' + snapshot_file)
        if not os.path.isfile(snapshot_file):
            print('cannot resume, no snapshots exist, check the code and your log directory for errors')
            exit(1)

    save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True

    # ------ Stacking Blocks and Grasping Specific Colors -----
    grasp_color_task = args.grasp_color_task
    place = args.place
    if grasp_color_task:
        if not is_sim:
            raise NotImplementedError('Real execution goal conditioning is not yet implemented')
        goal_condition_len = num_obj
    else:
        goal_condition_len = 0

    # Set random seed
    np.random.seed(random_seed)

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                  tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                  is_testing, test_preset_cases, test_preset_file, place, grasp_color_task)

    # Initialize trainer
    trainer = Trainer(method, push_rewards, future_reward_discount,
                      is_testing, snapshot_file, force_cpu,
                      goal_condition_len, place, pretrained, flops,
                      network=neural_network_name, common_sense=common_sense)

    if transfer_grasp_to_place:
        # Transfer pretrained grasp weights to the place action.
        trainer.model.transfer_grasp_to_place()

    # Initialize data logger
    title, dir_name = run_title(args)
    logger = Logger(continue_logging, logging_directory, args=args, dir_name=dir_name)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action': False,
                          'primitive_action': None,
                          'best_pix_ind': None,
                          'push_success': False,
                          'grasp_success': False,
                          'color_success': False,
                          'place_success': False,
                          'partial_stack_success': False,
                          'stack_height': 1,
                          'stack_rate': np.inf,
                          'trial_success_rate': np.inf,
                          'replay_iteration': 0,
                          'trial_complete': False,
                          'finalize_prev_trial_log': False,
                          'prev_stack_height': 1}

    # Do not save these nonlocal_variables or load them when resuming a run. They will be initialized to their default values
    always_default_nonlocals = ['executing_action',
                                'primitive_action']

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

        # this with block is skipped if the file doesn't exist
        nonlocal_vars_filename = os.path.join(logger.base_directory, 'data', 'nonlocal_vars.json')
        if os.path.exists(nonlocal_vars_filename):
            with open(nonlocal_vars_filename, 'r') as f:
                nonlocals_to_load = json.load(f)

                # in case not all nonlocal values were saved, only set what was saved
                for k, v in nonlocals_to_load.items():
                    if k not in always_default_nonlocals:
                        nonlocal_variables[k] = v

        num_trials = trainer.end_trial()
    else:
        num_trials = 0

    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = [2, 2] if not is_testing else [0, 0]
    explore_prob = 0.5 if not is_testing else 0.0

    if check_z_height:
        nonlocal_variables['stack_height'] = 0.0
        nonlocal_variables['prev_stack_height'] = 0.0
    best_stack_rate = np.inf

    if check_z_height:
        is_goal_conditioned = False
    else:
        is_goal_conditioned = grasp_color_task or place
    # Choose the first color block to grasp, or None if not running in goal conditioned mode
    if num_obj is not None:
        nonlocal_variables['stack'] = StackSequence(num_obj - num_extra_obj, is_goal_conditioned, trial=num_trials, total_steps=trainer.iteration)
    else:
        nonlocal_variables['stack'] = StackSequence(20, is_goal_conditioned, trial=num_trials, total_steps=trainer.iteration)

    num_trials = 0
    if continue_logging:
        num_trials = int(max(trainer.trial_log)[0])
        nonlocal_variables['stack'].trial = num_trials + 1

    if place:
        # If we are stacking we actually skip to the second block which needs to go on the first
        nonlocal_variables['stack'].next()

    def set_nonlocal_success_variables_false():
        nonlocal_variables['push_success'] = False
        nonlocal_variables['grasp_success'] = False
        nonlocal_variables['place_success'] = False
        nonlocal_variables['grasp_color_success'] = False
        nonlocal_variables['place_color_success'] = False

    def check_stack_update_goal(place_check=False, top_idx=-1, depth_img=None):
        """ Check nonlocal_variables for a good stack and reset if it does not match the current goal.

        # Params

            place_check: If place check is True we should match the current stack goal,
                all other actions should match the stack check excluding the top goal block,
                which will not have been placed yet.
            top_idx: The index of blocks sorted from high to low which is expected to contain the top stack block.
                -1 will be the highest object in the scene, -2 will be the second highest in the scene, etc.

        # Returns

        needed_to_reset boolean which is True if a reset was needed and False otherwise.
        """
        current_stack_goal = nonlocal_variables['stack'].current_sequence_progress()
        # no need to reset by default
        needed_to_reset = False
        if place_check:
            # Only reset while placing if the stack decreases in height!
            stack_shift = 1
        elif current_stack_goal is not None:
            # only the place check expects the current goal to be met
            current_stack_goal = current_stack_goal[:-1]
            stack_shift = 0
        # TODO(ahundt) BUG Figure out why a real stack of size 2 or 3 and a push which touches no blocks does not pass the stack_check and ends up a MISMATCH in need of reset. (update: may now be fixed, double check then delete when confirmed)
        if check_row:
            row_found, nonlocal_variables['stack_height'] = robot.check_row(current_stack_goal, num_obj=num_obj)
            # Note that for rows, a single action can make a row (horizontal stack) go from size 1 to a much larger number like 4.
            stack_matches_goal = nonlocal_variables['stack_height'] >= len(current_stack_goal)
        elif check_z_height:
            # decrease_threshold = None  # None means decrease_threshold will be disabled
            stack_matches_goal, nonlocal_variables['stack_height'], needed_to_reset = robot.check_z_height(depth_img, nonlocal_variables['prev_stack_height'])
            max_workspace_height = ' (see max_workspace_height printout above) '
            # TODO(ahundt) add a separate case for incremental height where continuous heights are converted back to height where 1.0 is the height of a block.
            # stack_matches_goal, nonlocal_variables['stack_height'] = robot.check_incremental_height(input_img, current_stack_goal)
        else:
            stack_matches_goal, nonlocal_variables['stack_height'] = robot.check_stack(current_stack_goal, top_idx=top_idx)
        nonlocal_variables['partial_stack_success'] = stack_matches_goal
        if not check_z_height:
            if nonlocal_variables['stack_height'] == 1:
                # A stack of size 1 does not meet the criteria for a partial stack success
                nonlocal_variables['partial_stack_success'] = False
                nonlocal_variables['stack_success'] = False
            max_workspace_height = len(current_stack_goal) - stack_shift
            # Has that stack gotten shorter than it was before? If so we need to reset
            needed_to_reset = nonlocal_variables['stack_height'] < max_workspace_height or nonlocal_variables['stack_height'] < nonlocal_variables['prev_stack_height']

        print('check_stack() stack_height: ' + str(nonlocal_variables['stack_height']) + ' stack matches current goal: ' + str(stack_matches_goal) + ' partial_stack_success: ' +
              str(nonlocal_variables['partial_stack_success']) + ' Does the code think a reset is needed: ' + str(needed_to_reset))
        # if place and needed_to_reset:
        # TODO(ahundt) BUG may reset push/grasp success too aggressively. If statement above and below for debugging, remove commented line after debugging complete
        if needed_to_reset or evaluate_random_objects:
            # we are two blocks off the goal, reset the scene.
            mismatch_str = 'main.py check_stack() DETECTED PROGRESS REVERSAL, mismatch between the goal height: ' + str(max_workspace_height) + ' and current workspace stack height: ' + str(nonlocal_variables['stack_height'])
            if not disable_situation_removal:
                mismatch_str += ', RESETTING the objects, goals, and action success to FALSE...'
            print(mismatch_str)
            if not disable_situation_removal:
                # this reset is appropriate for stacking, but not checking rows
                get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, trainer, '1')
                robot.reposition_objects()
                nonlocal_variables['stack'].reset_sequence()
                nonlocal_variables['stack'].next()
                # We needed to reset, so the stack must have been knocked over!
                # all rewards and success checks are False!
                set_nonlocal_success_variables_false()
                nonlocal_variables['trial_complete'] = True
                if check_row:
                    # on reset get the current row state
                    _, nonlocal_variables['stack_height'] = robot.check_row(current_stack_goal, num_obj=num_obj)
                    nonlocal_variables['prev_stack_height'] = nonlocal_variables['stack_height']
        return needed_to_reset

    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def process_actions():
        last_iteration_saved = -1  # used to prevent redundant saving

        action_count = 0
        grasp_count = 0
        successful_grasp_count = 0
        successful_color_grasp_count = 0
        place_count = 0
        place_rate = 0
        # short stacks of blocks
        partial_stack_count = 0
        partial_stack_rate = np.inf
        # all the blocks stacked
        stack_count = 0
        stack_rate = np.inf
        # will need to reset if something went wrong with stacking
        needed_to_reset = False
        grasp_str = ''
        successful_trial_count = int(np.max(trainer.trial_success_log)) if continue_logging and len(trainer.trial_success_log) > 0 else 0
        trial_rate = np.inf

        # when resuming a previous run, load variables from last saved iteration
        if continue_logging:
            process_vars = None
            resume_var_values_path = os.path.join(logger.base_directory, 'data', 'process_action_var_values.json')
            if os.path.exists(resume_var_values_path):
                with open(resume_var_values_path, 'r') as f:
                    process_vars = json.load(f)

                action_count = process_vars['action_count']
                grasp_count = process_vars['grasp_count']
                successful_grasp_count = process_vars['successful_grasp_count']
                successful_color_grasp_count = process_vars['successful_color_grasp_count']
                place_count = process_vars['place_count']
                place_rate = process_vars['place_rate']
                partial_stack_count = process_vars['partial_stack_count']
                partial_stack_rate = process_vars['partial_stack_rate']
                stack_count = process_vars['stack_count']
                stack_rate = process_vars['stack_rate']
                needed_to_reset = process_vars['needed_to_reset']
                grasp_str = process_vars['grasp_str']
                successful_trial_count = process_vars['successful_trial_count']
                trial_rate = process_vars['trial_rate']

            else:
                print("WARNING: missing process_action_var_values.json while resuming. Default values used")



        while True:
            if nonlocal_variables['executing_action']:
                action_count += 1
                # Determine whether grasping or pushing should be executed based on network predictions
                best_push_conf = np.ma.max(push_predictions)
                best_grasp_conf = np.ma.max(grasp_predictions)
                if place:
                    best_place_conf = np.ma.max(place_predictions)
                    print('Primitive confidence scores: %f (push), %f (grasp), %f (place)' % (best_push_conf, best_grasp_conf, best_place_conf))
                else:
                    print('Primitive confidence scores: %f (push), %f (grasp)' % (best_push_conf, best_grasp_conf))

                explore_actions = False
                # TODO(ahundt) this grasp/place condition needs refinement so we can do colors and grasp -> push -> place
                if place and nonlocal_variables['primitive_action'] == 'grasp' and nonlocal_variables['grasp_success']:
                    nonlocal_variables['primitive_action'] = 'place'
                else:
                    nonlocal_variables['primitive_action'] = 'grasp'

                if not grasp_only and not nonlocal_variables['primitive_action'] == 'place':
                    if is_testing and method == 'reactive':
                        if best_push_conf > 2 * best_grasp_conf:
                            nonlocal_variables['primitive_action'] = 'push'
                    else:
                        if best_push_conf > best_grasp_conf:
                            nonlocal_variables['primitive_action'] = 'push'
                    explore_actions = np.random.uniform() < explore_prob
                    # Exploitation (do best action) vs exploration (do random action)
                    if explore_actions:
                        print('Strategy: explore (exploration probability: %f)' % (explore_prob))
                        push_frequency_one_in_n = 5
                        nonlocal_variables['primitive_action'] = 'push' if np.random.randint(0, push_frequency_one_in_n) == 0 else 'grasp'
                    else:
                        print('Strategy: exploit (exploration probability: %f)' % (explore_prob))
                trainer.is_exploit_log.append([0 if explore_actions else 1])
                logger.write_to_log('is-exploit', trainer.is_exploit_log)
                # TODO(ahundt) remove if this has been working for a while, the trial log is now updated in the main thread rather than the robot control thread.
                # trainer.trial_log.append([nonlocal_variables['stack'].trial])
                # logger.write_to_log('trial', trainer.trial_log)

                # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
                each_action_max_coordinate = {
                    'push': np.unravel_index(np.ma.argmax(push_predictions), push_predictions.shape), # push, index 0
                    'grasp': np.unravel_index(np.ma.argmax(grasp_predictions), grasp_predictions.shape),
                }
                each_action_predicted_value = {
                    'push': push_predictions[each_action_max_coordinate['push']], # push, index 0
                    'grasp': grasp_predictions[each_action_max_coordinate['grasp']],
                }
                if place:
                    each_action_max_coordinate['place'] = np.unravel_index(np.ma.argmax(place_predictions), place_predictions.shape)
                    each_action_predicted_value['place'] = place_predictions[each_action_max_coordinate['place']]
                # we will actually execute the best pixel index of the selected action
                nonlocal_variables['best_pix_ind'] = each_action_max_coordinate[nonlocal_variables['primitive_action']]
                predicted_value = each_action_predicted_value[nonlocal_variables['primitive_action']]

                # If heuristic bootstrapping is enabled: if change has not been detected more than 2 times, execute heuristic algorithm to detect grasps/pushes
                # NOTE: typically not necessary and can reduce final performance.
                if heuristic_bootstrap and nonlocal_variables['primitive_action'] == 'push' and no_change_count[0] >= 2:
                    print('Change not detected for more than two pushes. Running heuristic pushing.')
                    nonlocal_variables['best_pix_ind'] = trainer.push_heuristic(valid_depth_heightmap)
                    no_change_count[0] = 0
                    predicted_value = push_predictions[nonlocal_variables['best_pix_ind']]
                    use_heuristic = True
                elif heuristic_bootstrap and nonlocal_variables['primitive_action'] == 'grasp' and no_change_count[1] >= 2:
                    print('Change not detected for more than two grasps. Running heuristic grasping.')
                    nonlocal_variables['best_pix_ind'] = trainer.grasp_heuristic(valid_depth_heightmap)
                    no_change_count[1] = 0
                    predicted_value = grasp_predictions[nonlocal_variables['best_pix_ind']]
                    use_heuristic = True
                else:
                    use_heuristic = False

                trainer.use_heuristic_log.append([1 if use_heuristic else 0])
                logger.write_to_log('use-heuristic', trainer.use_heuristic_log)

                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)

                # Compute 3D position of pixel
                print('Action: %s at (%d, %d, %d)' % (nonlocal_variables['primitive_action'], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(nonlocal_variables['best_pix_ind'][0]*(360.0/trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]

                # Adjust start position of all actions, and make sure z value is safe and not too low
                primitive_position, push_may_contact_something = action_heightmap_coordinate_to_3d_robot_pose(best_pix_x, best_pix_y, nonlocal_variables['primitive_action'])

                # Save executed primitive where [0, 1, 2] corresponds to [push, grasp, place]
                trainer.executed_action_log.append([ACTION_TO_ID[nonlocal_variables['primitive_action']], nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]])
                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    push_pred_vis = trainer.get_prediction_vis(push_predictions, color_heightmap, each_action_max_coordinate['push'])
                    logger.save_visualizations(trainer.iteration, push_pred_vis, 'push')
                    cv2.imwrite('visualization.push.png', push_pred_vis)
                    grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, each_action_max_coordinate['grasp'])
                    logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                    cv2.imwrite('visualization.grasp.png', grasp_pred_vis)
                    if place:
                        place_pred_vis = trainer.get_prediction_vis(place_predictions, color_heightmap, each_action_max_coordinate['place'])
                        logger.save_visualizations(trainer.iteration, place_pred_vis, 'place')
                        cv2.imwrite('visualization.place.png', place_pred_vis)

                # Initialize variables that influence reward
                set_nonlocal_success_variables_false()
                if place:
                    current_stack_goal = nonlocal_variables['stack'].current_sequence_progress()

                # Execute primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    if skip_noncontact_actions and not push_may_contact_something:
                        # We are too high to contact anything, don't bother actually pushing.
                        # TODO(ahundt) also check for case where we are too high for the local gripper path
                        nonlocal_variables['push_success'] = False
                    else:
                        nonlocal_variables['push_success'] = robot.push(primitive_position, best_rotation_angle, workspace_limits)

                    if place and check_row:
                        needed_to_reset = check_stack_update_goal()
                        if (not needed_to_reset and nonlocal_variables['partial_stack_success']):

                            if nonlocal_variables['stack_height'] >= len(current_stack_goal):
                                nonlocal_variables['stack'].next()
                                # TODO(ahundt) create a push to partial stack count separate from the place to partial stack count
                                partial_stack_count += 1
                            next_stack_goal = nonlocal_variables['stack'].current_sequence_progress()
                            if nonlocal_variables['stack_height'] >= nonlocal_variables['stack'].num_obj:
                                print('TRIAL ' + str(nonlocal_variables['stack'].trial) + ' SUCCESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                                nonlocal_variables['stack_success'] = True
                                stack_count += 1
                                # full stack complete! reset the scene
                                successful_trial_count += 1
                                get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, trainer, '1')
                                robot.reposition_objects()
                                if len(next_stack_goal) > 1:
                                    # if multiple parts of a row are completed in one action, we need to reset the trial counter.
                                    nonlocal_variables['stack'].reset_sequence()
                                # goal is 2 blocks in a row
                                nonlocal_variables['stack'].next()
                                nonlocal_variables['trial_complete'] = True

                    #TODO(hkwon214) Get image after executing push action. save also? better place to put?
                    valid_depth_heightmap_push, color_heightmap_push, depth_heightmap_push, color_img_push, depth_img_push = get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, trainer, save_image=False)
                    if place:
                        # Check if the push caused a topple, size shift zero because
                        # place operations expect increased height,
                        # while push expects constant height.
                        needed_to_reset = check_stack_update_goal(depth_img=valid_depth_heightmap_push)
                    if not place or not needed_to_reset:
                        print('Push motion successful (no crash, need not move blocks): %r' % (nonlocal_variables['push_success']))
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    grasp_count += 1
                    # TODO(ahundt) this probably will cause threading conflicts, add a mutex
                    if nonlocal_variables['stack'].object_color_index is not None and grasp_color_task:
                        grasp_color_name = robot.color_names[int(nonlocal_variables['stack'].object_color_index)]
                        print('Attempt to grasp color: ' + grasp_color_name)

                    if(skip_noncontact_actions and (np.isnan(valid_depth_heightmap[best_pix_y][best_pix_x]) or
                            valid_depth_heightmap[best_pix_y][best_pix_x] < 0.01)):
                        # Skip noncontact actions we don't bother actually grasping if there is nothing there to grasp
                        nonlocal_variables['grasp_success'], nonlocal_variables['grasp_color_success'] = False, False
                        print('Grasp action failure, heuristics determined grasp would not contact anything.')
                    else:
                        nonlocal_variables['grasp_success'], nonlocal_variables['grasp_color_success'] = robot.grasp(primitive_position, best_rotation_angle, object_color=nonlocal_variables['stack'].object_color_index)
                    print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))
                    # Get image after executing grasp action.
                    # TODO(ahundt) save also? better place to put?
                    valid_depth_heightmap_grasp, color_heightmap_grasp, depth_heightmap_grasp, color_img_grasp, depth_img_grasp = get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, trainer, save_image=False)
                    if place:
                        # when we are stacking we must also check the stack in case we caused it to topple
                        top_idx = -1
                        if nonlocal_variables['grasp_success']:
                            # we will need to check the second from top block for the stack
                            top_idx = -2
                        # check if a failed grasp led to a topple, or if the top block was grasped
                        # TODO(ahundt) in check_stack() support the check after a specific grasp in case of successful grasp topple. Perhaps allow the top block to be specified?
                        needed_to_reset = check_stack_update_goal(top_idx=top_idx, depth_img=valid_depth_heightmap_grasp)
                    if nonlocal_variables['grasp_success']:
                        # robot.restart_sim()
                        successful_grasp_count += 1
                        if grasp_color_task:
                            if nonlocal_variables['grasp_color_success']:
                                successful_color_grasp_count += 1
                            if not place:
                                # reposition the objects if we aren't also attempting to place correctly.
                                robot.reposition_objects()
                                nonlocal_variables['trial_complete'] = True

                            print('Successful color-specific grasp: %r intended target color: %s' % (nonlocal_variables['grasp_color_success'], grasp_color_name))
                    grasp_rate = float(successful_grasp_count) / float(grasp_count)
                    color_grasp_rate = float(successful_color_grasp_count) / float(grasp_count)
                    grasp_str = 'Grasp Count: %r, grasp success rate: %r' % (grasp_count, grasp_rate)
                    if grasp_color_task:
                        grasp_str += ' color success rate: %r' % (color_grasp_rate)
                    if not place:
                        print(grasp_str)
                elif nonlocal_variables['primitive_action'] == 'place':
                    place_count += 1
                    nonlocal_variables['place_success'] = robot.place(primitive_position, best_rotation_angle)

                    # Get image after executing place action.
                    # TODO(ahundt) save also? better place to put?
                    valid_depth_heightmap_place, color_heightmap_place, depth_heightmap_place, color_img_place, depth_img_place = get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, trainer, save_image=False)
                    needed_to_reset = check_stack_update_goal(place_check=True, depth_img=valid_depth_heightmap_place)
                    if not needed_to_reset and nonlocal_variables['place_success'] and nonlocal_variables['partial_stack_success']:
                        partial_stack_count += 1
                        # Only increment our progress checks if we've surpassed the current goal
                        # TODO(ahundt) check for a logic error between rows and stack modes due to if height ... next() check.
                        if not check_z_height and nonlocal_variables['stack_height'] >= len(current_stack_goal):
                            nonlocal_variables['stack'].next()
                        next_stack_goal = nonlocal_variables['stack'].current_sequence_progress()
                        if ((check_z_height and nonlocal_variables['stack_height'] > check_z_height_goal) or
                           (not check_z_height and len(next_stack_goal) < len(current_stack_goal))):
                            print('TRIAL ' + str(nonlocal_variables['stack'].trial) + ' SUCCESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                            nonlocal_variables['stack_success'] = True
                            stack_count += 1
                            # full stack complete! reset the scene
                            successful_trial_count += 1
                            get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, trainer, '1')
                            robot.reposition_objects()
                            # We don't need to reset here because the algorithm already reset itself
                            # nonlocal_variables['stack'].reset_sequence()
                            nonlocal_variables['stack'].next()
                            nonlocal_variables['trial_complete'] = True
                    # TODO(ahundt) perhaps reposition objects every time a partial stack step fails (partial_stack_success == false) to avoid weird states?

                trainer.grasp_success_log.append([int(nonlocal_variables['grasp_success'])])
                if grasp_color_task:
                    trainer.color_success_log.append([int(nonlocal_variables['color_success'])])
                if place:
                    # place trainer logs are updated in process_actions()
                    trainer.stack_height_log.append([float(nonlocal_variables['stack_height'])])
                    trainer.partial_stack_success_log.append([int(nonlocal_variables['partial_stack_success'])])
                    trainer.place_success_log.append([int(nonlocal_variables['place_success'])])
                    trainer.trial_success_log.append([int(successful_trial_count)])

                    if partial_stack_count > 0 and place_count > 0:
                        partial_stack_rate = float(action_count)/float(partial_stack_count)
                        place_rate = float(partial_stack_count)/float(place_count)
                    if stack_count > 0:
                        stack_rate = float(action_count)/float(stack_count)
                        nonlocal_variables['stack_rate'] = stack_rate
                        trial_rate = float(successful_trial_count)/float(nonlocal_variables['stack'].trial)
                        nonlocal_variables['trial_success_rate'] = trial_rate
                    print('STACK:  trial: ' + str(nonlocal_variables['stack'].trial) + ' actions/partial: ' + str(partial_stack_rate) +
                          '  actions/full stack: ' + str(stack_rate) +
                          ' (lower is better)  ' + grasp_str + ' place_on_stack_rate: ' + str(place_rate) + ' place_attempts: ' + str(place_count) +
                          '  partial_stack_successes: ' + str(partial_stack_count) +
                          '  stack_successes: ' + str(stack_count) + ' trial_success_rate: ' + str(trial_rate) + ' stack goal: ' + str(current_stack_goal) +
                          ' current_height: ' + str(nonlocal_variables['stack_height']))

                if check_z_height and nonlocal_variables['trial_complete']:
                    # Zero out the height because the trial is done.
                    # Note these lines must be after the logging of these variables is complete.
                    nonlocal_variables['stack_height'] = 0.0
                    nonlocal_variables['prev_stack_height'] = 0.0
                elif nonlocal_variables['trial_complete']:
                    # Set back to the minimum stack height because the trial is done.
                    # Note these lines must be after the logging of these variables is complete.
                    nonlocal_variables['stack_height'] = 1
                    nonlocal_variables['prev_stack_height'] = 1

                nonlocal_variables['executing_action'] = False

            # save this thread's variables every time the trial is completed (model is also saved at this time)
            if nonlocal_variables['trial_complete']:

                if last_iteration_saved != trainer.iteration: # checks if it already saved this iteration
                    last_iteration_saved = trainer.iteration

                    # create dict of all variables and save a json file
                    process_vars = {}
                    process_vars['action_count'] = action_count
                    process_vars['grasp_count'] = grasp_count
                    process_vars['successful_grasp_count'] = successful_grasp_count
                    process_vars['successful_color_grasp_count'] = successful_color_grasp_count
                    process_vars['place_count'] = place_count
                    process_vars['place_rate'] = place_rate
                    process_vars['partial_stack_count'] = partial_stack_count
                    process_vars['partial_stack_rate'] = partial_stack_rate
                    process_vars['stack_count'] = stack_count
                    process_vars['stack_rate'] = stack_rate
                    process_vars['needed_to_reset'] = needed_to_reset
                    process_vars['grasp_str'] = grasp_str
                    process_vars['successful_trial_count'] = successful_trial_count
                    process_vars['trial_rate'] = trial_rate

                    with open(os.path.join(logger.base_directory, 'data', 'process_action_var_values.json'), 'w') as f:
                            json.dump(process_vars, f)


            # TODO(ahundt) this should really be using proper threading and locking algorithms
            time.sleep(0.01)

    def action_heightmap_coordinate_to_3d_robot_pose(best_pix_x, best_pix_y, action_name, robot_push_vertical_offset=0.026):
        # Adjust start position of all actions, and make sure z value is safe and not too low
        def get_local_region(heightmap, region_width=0.03):
            safe_kernel_width = int(np.round((region_width/2)/heightmap_resolution))
            return heightmap[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1, heightmap.shape[0]), max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, heightmap.shape[1])]
        # make sure the fingers will not collide with the objects
        finger_width = 0.04
        finger_touchdown_region = get_local_region(valid_depth_heightmap, region_width=finger_width)
        safe_z_position = workspace_limits[2][0]
        if finger_touchdown_region.size != 0:
            safe_z_position += np.max(finger_touchdown_region)
        else:
            safe_z_position += valid_depth_heightmap[best_pix_y][best_pix_x]
        if robot.background_heightmap is not None:
            # add the height of the background scene
            safe_z_position += np.max(get_local_region(robot.background_heightmap, region_width=0.03))
        push_may_contact_something = False
        if action_name == 'push':
            # determine if the safe z position might actually contact anything during the push action
            # TODO(ahundt) common sense push motion region can be refined based on the rotation angle and the direction of travel
            push_width = 0.2
            local_push_region = get_local_region(valid_depth_heightmap, region_width=push_width)
            # push_may_contact_something is True for something noticeably higher than the push action z height
            max_local_push_region = np.max(local_push_region)
            if max_local_push_region < 0.01:
                # if there is nothing more than 1cm tall, there is nothing to push
                push_may_contact_something = False
            else:
                push_may_contact_something = safe_z_position - workspace_limits[2][0] + robot_push_vertical_offset < max_local_push_region
            # print('>>>> Gripper will push at height: ' + str(safe_z_position) + ' max height of stuff: ' + str(max_local_push_region) + ' predict contact: ' + str(push_may_contact_something))
            push_str = ''
            if not push_may_contact_something:
                push_str += 'Predicting push action failure, heuristics determined '
                push_str += 'push at height ' + str(safe_z_position)
                push_str += ' would not contact anything at the max height of ' + str(max_local_push_region)
                print(push_str)

        primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0], best_pix_y * heightmap_resolution + workspace_limits[1][0], safe_z_position]
        return primitive_position, push_may_contact_something

    # TODO(ahundt) create a new experience replay reward schedule that goes backwards across multiple time steps.

    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    prev_primitive_action = None
    prev_reward_value = None
    if test_preset_cases:
        # save out the order we will visit the preset files for a sanity check
        print('preset files order: ' + str(preset_files))
        np.savetxt(os.path.join(logger.transitions_directory, 'preset-case-files.log.txt'), preset_files, delimiter=' ', fmt='%s')
    if show_preset_cases_then_exit and test_preset_cases:
        # Just a quick temporary mode for viewing the saved preset test cases
        for case_file in preset_files:
            # load the current preset case, incrementing as trials are cleared
            print('loading case file: ' + str(case_file))
            robot.load_preset_case(case_file)
            robot.restart_sim()
            robot.add_objects()
            time.sleep(3)
        exit_called = True
        robot.shutdown()
        return

    num_trials = trainer.num_trials()
    do_continue = False
    # Start main training/testing loop, max_iter == 0 or -1 goes forever.
    while max_iter < 0 or trainer.iteration < max_iter:
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()
        # Record the current trial number
        trainer.trial_log.append([trainer.num_trials()])

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim:
            robot.check_sim()

        # Get latest RGB-D image
        valid_depth_heightmap, color_heightmap, depth_heightmap, color_img, depth_img = get_and_save_images(
            robot, workspace_limits, heightmap_resolution, logger, trainer)

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.02] = 1
        if show_heightmap:
            # show the heightmap
            f = plt.figure()
            f.suptitle(str(trainer.iteration))
            f.add_subplot(1,3, 1)
            plt.imshow(valid_depth_heightmap)
            f.add_subplot(1,3, 2)
            # f.add_subplot(1,2, 1)
            if robot.background_heightmap is not None:
                plt.imshow(robot.background_heightmap)
                f.add_subplot(1,3, 3)
            plt.imshow(stuff_count)
            plt.show(block=True)
        stuff_sum = np.sum(stuff_count)
        empty_threshold = 300
        if is_sim and is_testing:
            empty_threshold = 10
        if check_row:
            # here we are assuming blocks for check_row, if any block leaves the scene then we can't succeed.
            # TODO(ahundt) Ideally volume should also be accounted for, a perfect stack is about the area of 1 block, and the scene might start with a stack.
            num_empty_obj = num_obj
            if is_testing:
                num_empty_obj -= 1
            empty_threshold = 300 * (num_empty_obj + num_extra_obj)
        print('Current count of pixels with stuff: ' + str(stuff_sum) + ' threshold below which the scene is considered empty: ' + str(empty_threshold))
        if not place and stuff_sum < empty_threshold:
            print('Pushing And Grasping Trial Successful!')
            num_trials = trainer.num_trials()
            pg_trial_success_count = np.max(trainer.trial_success_log, initial=0)
            for i in range(len(trainer.trial_success_log), num_trials):
                # previous trials were ended early
                trainer.trial_success_log.append([int(pg_trial_success_count)])
            trainer.trial_success_log.append([int(pg_trial_success_count + 1)])
            nonlocal_variables['trial_complete'] = True

        if stuff_sum < empty_threshold or (is_sim and no_change_count[0] + no_change_count[1] > 10):
            if is_sim:
                print('There have not been changes to the objects for for a long time [push, grasp]: ' + str(no_change_count) +
                      ', or there are not enough objects in view (value: %d)! Repositioning objects.' % (stuff_sum))
                robot.restart_sim()
                robot.add_objects()
                if is_testing:  # If at end of test run, re-load original weights (before test run)
                    trainer.model.load_state_dict(torch.load(snapshot_file))
                if place:
                    set_nonlocal_success_variables_false()
                    nonlocal_variables['stack'].reset_sequence()
                    nonlocal_variables['stack'].next()
            else:
                # print('Not enough stuff on the table (value: %d)! Pausing for 30 seconds.' % (np.sum(stuff_count)))
                # time.sleep(30)
                print('Not enough stuff on the table (value: %d)! Flipping over bin of objects...' % (stuff_sum))
                robot.restart_real()

            nonlocal_variables['trial_complete'] = True
            # TODO(ahundt) might this continue statement increment trainer.iteration, break accurate indexing of the clearance log into the label, reward, and image logs?
            do_continue = True
            # continue

        if nonlocal_variables['trial_complete']:
            # Check if the other thread ended the trial and reset the important values
            no_change_count = [0, 0]
            num_trials = trainer.end_trial()
            logger.write_to_log('clearance', trainer.clearance_log)
            # we've recorded the data to mark this trial as complete
            nonlocal_variables['trial_complete'] = False
            # we're still not totally done, we still need to finilaize the log for the trial
            nonlocal_variables['finalize_prev_trial_log'] = True
            if is_testing and test_preset_cases:
                case_file = preset_files[min(len(preset_files)-1, int(float(num_trials+1)/float(trials_per_case)))]
                # case_file = preset_files[min(len(preset_files)-1, int(float(num_trials-1)/float(trials_per_case)))]
                # load the current preset case, incrementing as trials are cleared
                print('loading case file: ' + str(case_file))
                robot.load_preset_case(case_file)
            if is_testing and not place and num_trials >= max_test_trials:
                exit_called = True  # Exit after training thread (backprop and saving labels)
            if do_continue:
                do_continue = False
                continue

            # TODO(ahundt) update experience replay trial rewards

        # check for possible bugs in the code
        if len(trainer.reward_value_log) < trainer.iteration - 2:
            # check for progress counting inconsistencies
            print('WARNING POSSIBLE CRITICAL ERROR DETECTED: log data index and trainer.iteration out of sync!!! Experience Replay may break! '
                  'Check code for errors in indexes, continue statements etc.')
        if place and nonlocal_variables['stack'].trial != num_trials + 1:
            # check that num trials is always 1 less than the current trial number
            print('WARNING variable mismatch num_trials + 1: ' + str(num_trials + 1) + ' nonlocal_variables[stack].trial: ' + str(nonlocal_variables['stack'].trial))

        # check if we have completed the current test
        if is_testing and place and nonlocal_variables['stack'].trial > max_test_trials:
            # If we are doing a fixed number of test trials, end the run the next time around.
            exit_called = True

        if not exit_called:

            # Run forward pass with network to get affordances
            if nonlocal_variables['stack'].is_goal_conditioned_task and grasp_color_task:
                goal_condition = np.array([nonlocal_variables['stack'].current_one_hot()])
            else:
                goal_condition = None

            push_predictions, grasp_predictions, place_predictions, state_feat, output_prob = trainer.forward(
                color_heightmap, valid_depth_heightmap, is_volatile=True, goal_condition=goal_condition)

            # Execute best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' in locals():

            # Detect changes
            depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff > 0.3] = 0
            depth_diff[depth_diff < 0.01] = 0
            depth_diff[depth_diff > 0] = 1
            # NOTE: original VPG change_threshold was 300
            change_threshold = 300
            change_value = np.sum(depth_diff)
            change_detected = change_value > change_threshold or prev_grasp_success
            print('Change detected: %r (value: %d)' % (change_detected, change_value))

            if change_detected:
                if prev_primitive_action == 'push':
                    no_change_count[0] = 0
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] = 0
            else:
                if prev_primitive_action == 'push':
                    no_change_count[0] += 1
                elif prev_primitive_action == 'grasp':
                    no_change_count[1] += 1

            if no_height_reward:
                # used to assess the value of the reward multiplier
                reward_multiplier = 1
            else:
                reward_multiplier = prev_stack_height

            # Compute training labels, returns are:
            # label_value == expected_reward (with future rewards)
            # prev_reward_value == current_reward (without future rewards)
            label_value, prev_reward_value = trainer.get_label_value(
                prev_primitive_action, prev_push_success, prev_grasp_success, change_detected,
                prev_push_predictions, prev_grasp_predictions, color_heightmap, valid_depth_heightmap,
                prev_color_success, goal_condition=prev_goal_condition, prev_place_predictions=prev_place_predictions,
                place_success=prev_partial_stack_success, reward_multiplier=reward_multiplier)
            # label_value is also known as expected_reward in trainer.get_label_value(), this is what the nn predicts.
            trainer.label_value_log.append([label_value])
            logger.write_to_log('label-value', trainer.label_value_log)
            # prev_reward_value is the regular old reward value actually based on the multiplier and action success
            trainer.reward_value_log.append([prev_reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)
            trainer.change_detected_log.append([change_detected])
            logger.write_to_log('change-detected', trainer.change_detected_log)
            logger.write_to_log('grasp-success', trainer.grasp_success_log)
            if nonlocal_variables['stack'].is_goal_conditioned_task and grasp_color_task:
                trainer.goal_condition_log.append(nonlocal_variables['stack'].current_one_hot())
                logger.write_to_log('goal-condition', trainer.goal_condition_log)
                logger.write_to_log('color-success', trainer.color_success_log)
            if place:
                logger.write_to_log('stack-height', trainer.stack_height_log)
                logger.write_to_log('partial-stack-success', trainer.partial_stack_success_log)
                logger.write_to_log('place-success', trainer.place_success_log)
            if nonlocal_variables['finalize_prev_trial_log']:
                # Do final logging from the previous trial and previous complete iteration
                nonlocal_variables['finalize_prev_trial_log'] = False
                trainer.trial_reward_value_log_update()
                logger.write_to_log('trial-reward-value', trainer.trial_reward_value_log)
                logger.write_to_log('iteration', np.array([trainer.iteration]))
                logger.write_to_log('trial-success', trainer.trial_success_log)
                logger.write_to_log('trial', trainer.trial_log)
                if trainer.iteration > 1000:
                    plot.plot_it(logger.base_directory, title, place=place)
                print('Trial logging complete: ' + str(num_trials) + ' --------------------------------------------------------------')

            # Backpropagate
            if not disable_two_step_backprop:
                trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action, prev_best_pix_ind, label_value, goal_condition=prev_goal_condition)

            # Adjust exploration probability
            if not is_testing:
                explore_prob = max(0.5 * np.power(0.9998, trainer.iteration), 0.1) if explore_rate_decay else 0.5

            # Do sampling for experience replay
            if experience_replay_enabled and prev_reward_value is not None and not is_testing:
                # Here we will try to sample a reward value from the same action as the current one
                # which differs from the most recent reward value to reduce the chance of catastrophic forgetting.
                # TODO(ahundt) experience replay is very hard-coded with lots of bugs, won't evaluate all reward possibilities, and doesn't deal with long range time dependencies.
                experience_replay(method, prev_primitive_action, prev_reward_value, trainer, grasp_color_task, logger, nonlocal_variables, place, goal_condition, trial_reward=trial_reward)

            # latest model and best model are stored
            # Save model snapshot
            if not is_testing:
                logger.save_backup_model(trainer.model, method)
                if nonlocal_variables['trial_complete']:  # saves once every time a trial is completed
                    logger.save_model(trainer.model, method)

                    # copy nonlocal_variable values and discard those which should be default when resuming.
                    nonlocals_to_save = nonlocal_variables.copy()
                    entries_to_pop = always_default_nonlocals.copy()

                    # save all entries which are JSON serializable only. Otherwise don't save
                    for k, v in nonlocals_to_save.items():
                        if not utils.is_jsonable(v):
                            entries_to_pop.append(k)

                    for k in entries_to_pop:
                        nonlocals_to_save.pop(k)

                    print('################### LOGGING DIR', logger.base_directory)
                    with open(os.path.join(logger.base_directory, 'data', 'nonlocal_vars.json'), 'w') as f:
                        json.dump(nonlocals_to_save, f)

                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()

                # Save model if we are at a new best stack rate
                if place and trainer.iteration >= 1000 and nonlocal_variables['stack_rate'] < best_stack_rate:
                    best_stack_rate = nonlocal_variables['stack_rate']
                    stack_rate_str = method + '-best-stack-rate'
                    logger.save_backup_model(trainer.model, stack_rate_str)
                    logger.save_model(trainer.model, stack_rate_str)
                    logger.write_to_log('best-iteration', np.array([trainer.iteration]))

                    # copy nonlocal_variable values and discard those which should be default when resuming.
                    nonlocals_to_save = nonlocal_variables.copy()
                    entries_to_pop = always_default_nonlocals.copy()

                    # save all entries which are JSON serializable only. Otherwise don't save
                    for k, v in nonlocals_to_save.items():
                        if not utils.is_jsonable(v):
                            entries_to_pop.append(k)

                    for k in entries_to_pop:
                        nonlocals_to_save.pop(k)

                    with open(os.path.join(logger.base_directory, 'data', 'best_nonlocal_vars.json'), 'w') as f:
                        json.dump(nonlocals_to_save, f)

                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()

        # Sync both action thread and training thread
        num_problems_detected = 0
        while nonlocal_variables['executing_action']:
            if experience_replay_enabled and prev_reward_value is not None and not is_testing:
                # do some experience replay while waiting, rather than sleeping
                experience_replay(method, prev_primitive_action, prev_reward_value, trainer, grasp_color_task, logger, nonlocal_variables, place, goal_condition, trial_reward=trial_reward)
            else:
                time.sleep(0.1)
            time_elapsed = time.time()-iteration_time_0
            if int(time_elapsed) > 25:
                # TODO(ahundt) double check that this doesn't screw up state completely for future trials...
                print('ERROR: PROBLEM DETECTED IN SCENE, NO CHANGES FOR OVER 25 SECONDS, RESETTING THE OBJECTS TO RECOVER...')
                get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, trainer, '1')
                if is_sim:
                    robot.check_sim()
                    if not robot.reposition_objects():
                        # This can happen if objects are in impossible positions (NaN),
                        # so set the variable to immediately and completely restart
                        # the simulation below.
                        num_problems_detected += 3
                nonlocal_variables['trial_complete'] = True
                if place:
                    nonlocal_variables['stack'].reset_sequence()
                    nonlocal_variables['stack'].next()
                if check_z_height:
                    # Zero out the height because the trial is done.
                    # Note these lines must normally be after the
                    # logging of these variables is complete,
                    # but this is a special (hopefully rare) recovery scenario.
                    nonlocal_variables['stack_height'] = 0.0
                    nonlocal_variables['prev_stack_height'] = 0.0
                else:
                    nonlocal_variables['stack_height'] = 1.0
                    nonlocal_variables['prev_stack_height'] = 1.0
                num_problems_detected += 1
                if num_problems_detected > 2 and is_sim:
                    # Try more drastic recovery methods the second time around
                    robot.restart_sim(connect=True)
                    robot.add_objects()
                # don't reset again for 20 more seconds
                iteration_time_0 = time.time()
                # TODO(ahundt) Improve recovery: maybe set trial_complete = True here and call continue or set do_continue = True?

        if exit_called:
            # shut down the simulation or robot
            robot.shutdown()
            break

        # Save information for next training step
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_push_success = nonlocal_variables['push_success']
        prev_grasp_success = nonlocal_variables['grasp_success']
        prev_primitive_action = nonlocal_variables['primitive_action']
        prev_place_success = nonlocal_variables['place_success']
        prev_partial_stack_success = nonlocal_variables['partial_stack_success']
        # stack_height will just always be 1 if we are not actually stacking
        prev_stack_height = nonlocal_variables['stack_height']
        nonlocal_variables['prev_stack_height'] = nonlocal_variables['stack_height']
        prev_push_predictions = push_predictions.copy()
        prev_grasp_predictions = grasp_predictions.copy()
        prev_place_predictions = place_predictions
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']
        # TODO(ahundt) BUG We almost certainly need to copy nonlocal_variables['stack']
        prev_stack = nonlocal_variables['stack']
        prev_goal_condition = goal_condition
        if grasp_color_task:
            prev_color_success = nonlocal_variables['grasp_color_success']
            if nonlocal_variables['grasp_success'] and nonlocal_variables['grasp_color_success']:
                # Choose the next color block to grasp, or None if not running in goal conditioned mode
                nonlocal_variables['stack'].next()
                print('NEW GOAL COLOR: ' + str(robot.color_names[nonlocal_variables['stack'].object_color_index]) + ' GOAL CONDITION ENCODING: ' + str(nonlocal_variables['stack'].current_one_hot()))
        else:
            prev_color_success = None

        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1-iteration_time_0))

        print('Trainer iteration: %f' % (trainer.iteration))

def get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, trainer, filename_poststring='0', save_image=True):
    # Get latest RGB-D image
    color_img, depth_img = robot.get_camera_data()
    depth_img = depth_img * robot.cam_depth_scale  # Apply depth scale from calibration
    # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
    color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose,
                                                           workspace_limits, heightmap_resolution, background_heightmap=robot.background_heightmap)
    # TODO(ahundt) switch to masked array, then only have a regular heightmap
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

    # Save RGB-D images and RGB-D heightmaps
    if save_image:
        logger.save_images(trainer.iteration, color_img, depth_img, filename_poststring)
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, filename_poststring)
    return valid_depth_heightmap, color_heightmap, depth_heightmap, color_img, depth_img

def experience_replay(method, prev_primitive_action, prev_reward_value, trainer, grasp_color_task, logger, nonlocal_variables, place, goal_condition, all_history_prob=0.05, trial_reward=False):
    # Here we will try to sample a reward value from the same action as the current one
    # which differs from the most recent reward value to reduce the chance of catastrophic forgetting.
    # TODO(ahundt) experience replay is very hard-coded with lots of bugs, won't evaluate all reward possibilities, and doesn't deal with long range time dependencies.
    sample_primitive_action = prev_primitive_action
    sample_primitive_action_id = ACTION_TO_ID[sample_primitive_action]
    if trial_reward and len(trainer.trial_reward_value_log) > 2:
        max_iteration = len(trainer.trial_reward_value_log)
    else:
        trial_reward = False
        max_iteration = trainer.iteration
    # executed_action_log includes the action, push grasp or place, and the best pixel index
    actions = np.asarray(trainer.executed_action_log)[1:max_iteration, 0]
    prev_success = np.array(bool(prev_reward_value))

    # Get samples of the same primitive but with different success results
    if np.random.random(1) < all_history_prob:
        # Sample all of history every one out of n times.
        sample_ind = np.arange(1, max_iteration-1).reshape(max_iteration-2, 1)
    elif sample_primitive_action == 'push':
        # sample_primitive_action_id = 0
        sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.change_detected_log)[1:max_iteration, 0] != prev_success,
                                                actions == sample_primitive_action_id))
    elif sample_primitive_action == 'grasp':
        # sample_primitive_action_id = 1
        sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.grasp_success_log)[1:max_iteration, 0] != prev_success,
                                                actions == sample_primitive_action_id))
    elif sample_primitive_action == 'place':
        sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.partial_stack_success_log)[1:max_iteration, 0] != prev_success,
                                                actions == sample_primitive_action_id))
    else:
        raise NotImplementedError('ERROR: ' + sample_primitive_action + ' action is not yet supported in experience replay')

    if sample_ind.size == 0 and prev_reward_value is not None and max_iteration > 2:
        print('Experience Replay: We do not have samples for the ' + sample_primitive_action + ' action with a success state of ' + str(not prev_success) + ', so sampling from the whole history.')
        sample_ind = np.arange(1, max_iteration-1).reshape(max_iteration-2, 1)

    if sample_ind.size > 0:
        # Find sample with highest surprise value
        if method == 'reactive':
            # TODO(ahundt) BUG what to do with prev_reward_value? (formerly named sample_reward_value in previous commits)
            sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:, 0]] - (1 - prev_reward_value))
        elif method == 'reinforcement':
            sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:, 0]] - np.asarray(trainer.label_value_log)[sample_ind[:,0]])
        sorted_surprise_ind = np.argsort(sample_surprise_values[:, 0])
        sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
        pow_law_exp = 2
        rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
        # sample_iteration is the actual time step on which we will run experience replay
        sample_iteration = sorted_sample_ind[rand_sample_ind]

        nonlocal_variables['replay_iteration'] += 1
        # Load the data from disk, and run a forward pass with the current model
        [sample_stack_height, sample_primitive_action_id, sample_grasp_success,
         sample_change_detected, sample_push_predictions, sample_grasp_predictions,
         next_sample_color_heightmap, next_sample_depth_heightmap, sample_color_success,
         exp_goal_condition, sample_place_predictions, sample_place_success, sample_color_heightmap,
         sample_depth_heightmap] = trainer.load_sample(sample_iteration, logger)

        sample_primitive_action = ID_TO_ACTION[sample_primitive_action_id]
        print('Experience replay %d: history timestep index %d, action: %s, surprise value: %f' % (nonlocal_variables['replay_iteration'], sample_iteration, str(sample_primitive_action), sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))
        # sample_push_success is always true in the current version, because it only checks if the push action run, not if something was actually pushed, that is handled by change_detected.
        sample_push_success = True
        # TODO(ahundt) deletme if this has been working for a while, sample reward value isn't actually used for anything...
        # if trial_reward:
        #     sample_reward_value = trainer.trial_reward_value_log[sample_iteration]
        # else:
        #     sample_reward_value = trainer.reward_value_log[sample_iteration]

        # if no_height_reward:  # TODO(ahundt) why does the args.no_height_reward line below work and the regular no_height_reward here broken?
        if args.no_height_reward:
            # used to assess the value of the reward multiplier
            reward_multiplier = 1
        else:
            reward_multiplier = sample_stack_height
        # TODO(ahundt) This mix of current and next parameters (like next_sample_color_heightmap and sample_push_success) seems a likely spot for a bug, we must make sure we haven't broken the behavior. ahundt has already fixed one bug here.
        # get_label_value does the forward pass for updating the label value log.
        update_label_value_log = False
        if update_label_value_log:
            new_sample_label_value, _ = trainer.get_label_value(
                sample_primitive_action, sample_push_success, sample_grasp_success, sample_change_detected,
                sample_push_predictions, sample_grasp_predictions, next_sample_color_heightmap, next_sample_depth_heightmap,
                sample_color_success, goal_condition=exp_goal_condition, prev_place_predictions=sample_place_predictions,
                place_success=sample_place_success, reward_multiplier=reward_multiplier)

        if trial_reward:
            reward_to_backprop = trainer.trial_reward_value_log[sample_iteration]
        else:
            reward_to_backprop = trainer.label_value_log[sample_iteration]

        # Get labels for sample and backpropagate, trainer.backprop also does a forward pass internally.
        sample_best_pix_ind = np.asarray(trainer.executed_action_log)[sample_iteration, 1:4].astype(np.int)
        trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action, sample_best_pix_ind,
                         reward_to_backprop, goal_condition=exp_goal_condition)
        # Recompute prediction value and label for replay buffer
        if sample_primitive_action == 'push':
            trainer.predicted_value_log[sample_iteration] = [np.ma.max(sample_push_predictions)]
            # trainer.predicted_value_log[sample_iteration] = [sample_push_predictions[sample_best_pix_ind[0], sample_best_pix_ind[1], sample_best_pix_ind[2]]]
        elif sample_primitive_action == 'grasp':
            trainer.predicted_value_log[sample_iteration] = [np.ma.max(sample_grasp_predictions)]
            # trainer.predicted_value_log[sample_iteration] = [sample_grasp_predictions[sample_best_pix_ind[0], sample_best_pix_ind[1], sample_best_pix_ind[2]]]
        elif sample_primitive_action == 'place':
            trainer.predicted_value_log[sample_iteration] = [np.ma.max(sample_place_predictions)]
            # trainer.predicted_value_log[sample_iteration] = [sample_place_predictions[sample_best_pix_ind[0], sample_best_pix_ind[1], sample_best_pix_ind[2]]]

        if update_label_value_log:
            trainer.label_value_log[sample_iteration] = [new_sample_label_value]

    else:
        # print('Experience Replay: 0 prior training samples. Skipping experience replay.')
        time.sleep(0.01)



if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing, grasping, and placing as well as multi-step tasks for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                    help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,                                help='number of objects to add to simulation')
    parser.add_argument('--num_extra_obj', dest='num_extra_obj', type=int, action='store', default=0,                     help='number of secondary objects, like distractors, to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='192.168.1.155',                     help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,                           help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='192.168.1.155',                     help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,                           help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')
    parser.add_argument('--flops', dest='flops', action='store_true', default=False,                                      help='calculate floating point operations of a forward pass then exit')
    parser.add_argument('--show_heightmap', dest='show_heightmap', action='store_true', default=False,                    help='show the background heightmap for collecting a new one and debugging')

    # ------------- Algorithm options -------------
    parser.add_argument('--method', dest='method', action='store', default='reinforcement',                               help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
    parser.add_argument('--push_rewards', dest='push_rewards', action='store_true', default=False,                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,              help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,          help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    parser.add_argument('--grasp_only', dest='grasp_only', action='store_true', default=False)
    parser.add_argument('--check_row', dest='check_row', action='store_true', default=False,                              help='check for placed rows instead of stacks')
    parser.add_argument('--random_weights', dest='random_weights', action='store_true', default=False,                    help='use random weights rather than weights pretrained on ImageNet')
    parser.add_argument('--max_iter', dest='max_iter', action='store', type=int, default=-1,                              help='max iter for training. -1 (default) trains indefinitely.')
    parser.add_argument('--place', dest='place', action='store_true', default=False,                                      help='enable placing of objects')
    parser.add_argument('--skip_noncontact_actions', dest='skip_noncontact_actions', action='store_true', default=False,  help='enable skipping grasp and push actions when the heightmap is zero')
    parser.add_argument('--common_sense', dest='common_sense', action='store_true', default=False,                        help='Use common sense heuristics to detect and train on regions which do not contact anything, and will thus not result in task progress.')
    parser.add_argument('--no_height_reward', dest='no_height_reward', action='store_true', default=False,                help='disable stack height reward multiplier')
    parser.add_argument('--grasp_color_task', dest='grasp_color_task', action='store_true', default=False,                help='enable grasping specific colored objects')
    parser.add_argument('--grasp_count', dest='grasp_cout', type=int, action='store', default=0,                          help='number of successful task based grasps')
    parser.add_argument('--transfer_grasp_to_place', dest='transfer_grasp_to_place', action='store_true', default=False,  help='Load the grasping weights as placing weights.')
    parser.add_argument('--check_z_height', dest='check_z_height', action='store_true', default=False,                    help='use check_z_height instead of check_stacks for any stacks')
    # TODO(ahundt) determine a way to deal with the side effect
    parser.add_argument('--trial_reward', dest='trial_reward', action='store_true', default=False,                        help='Experience replay delivers rewards for the whole trial, not just next step. ')
    parser.add_argument('--disable_two_step_backprop', dest='disable_two_step_backprop', action='store_true', default=False,                        help='There is a local two time step training and backpropagation which does not precisely match trial rewards, this flag disables it. ')
    parser.add_argument('--check_z_height_goal', dest='check_z_height_goal', action='store', type=float, default=4.0,          help='check_z_height goal height, a value of 2.0 is 0.1 meters, and a value of 4.0 is 0.2 meters')
    parser.add_argument('--disable_situation_removal', dest='disable_situation_removal', action='store_true', default=False,                        help='Disables situation removal, where rewards are set to 0 and a reset is triggerd upon reveral of task progress. ')

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--evaluate_random_objects', dest='evaluate_random_objects', action='store_true', default=False,                help='Evaluate trials with random block positions, for example testing frequency of random rows.')
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=100,                help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='')
    parser.add_argument('--test_preset_dir', dest='test_preset_dir', action='store', default='simulation/test-cases/')
    parser.add_argument('--show_preset_cases_then_exit', dest='show_preset_cases_then_exit', action='store_true', default=False,    help='just show all the preset cases so you can have a look, then exit')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store', default='',                              help='snapshot file to load for the model')
    parser.add_argument('--nn', dest='nn', action='store', default='densenet',                                            help='Neural network architecture choice, options are efficientnet, densenet')
    parser.add_argument('--resume', dest='resume', nargs='?', default=None, const='last',                                 help='resume a previous run. If no run specified, resumes the most recent')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
