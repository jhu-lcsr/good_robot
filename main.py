#!/usr/bin/env python

import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
import torch
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils

# to convert action names to the corresponding ID number and vice-versa
ACTION_TO_ID = {'push':0, 'grasp':1, 'place':2}
ID_TO_ACTION = {0:'push', 1:'grasp', 2:'place'}

class StackSequence(object):
    def __init__(self, num_obj, is_goal_conditioned_task=True):
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
        self.trial = 0
        self.reset_sequence()
        self.total_steps = 1

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
        return self.object_color_sequence[:self.object_color_index+1]

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
    tcp_port = args.tcp_port if not is_sim else None
    rtc_host_ip = args.rtc_host_ip if not is_sim else None # IP and port to robot arm as real-time client (UR5)
    rtc_port = args.rtc_port if not is_sim else None
    if is_sim:
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    else:
        workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    heightmap_resolution = args.heightmap_resolution # Meters per pixel of heightmap
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # ------------- Algorithm options -------------
    method = args.method # 'reactive' (supervised learning) or 'reinforcement' (reinforcement learning ie Q-learning)
    push_rewards = args.push_rewards if method == 'reinforcement' else None  # Use immediate rewards (from change detection) for pushing?
    future_reward_discount = args.future_reward_discount
    experience_replay_enabled = args.experience_replay # Use prioritized experience replay?
    heuristic_bootstrap = args.heuristic_bootstrap # Use handcrafted grasping algorithm when grasping fails too many times in a row?
    explore_rate_decay = args.explore_rate_decay
    grasp_only = args.grasp_only
    pretrained = not args.random_weights
    max_iter = args.max_iter
    no_height_reward = args.no_height_reward

    # -------------- Test grasp_onlying options --------------
    is_testing = args.is_testing
    max_test_trials = args.max_test_trials # Maximum number of test runs per case/scenario
    test_preset_cases = args.test_preset_cases
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file)  if load_snapshot else None
    continue_logging = args.continue_logging # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True

    # ------ HK: Added Options -----
    grasp_color_task = args.grasp_color_task
    place = args.place
    if grasp_color_task:
        if not is_sim:
            raise NotImplementedError('Real execution goal conditioning is not yet implemented')
        goal_condition_len = num_obj
    else:
        goal_condition_len = 0
    #grasp_count = args.grasp_count

    # Set random seed
    np.random.seed(random_seed)

    # Initialize pick-and-place system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                  tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                  is_testing, test_preset_cases, test_preset_file, place, grasp_color_task)

    # Initialize trainer
    trainer = Trainer(method, push_rewards, future_reward_discount,
                      is_testing, load_snapshot, snapshot_file, force_cpu, goal_condition_len, place, pretrained)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters

    # Find last executed iteration of pre-loaded log, and load execution info and RL variables
    if continue_logging:
        trainer.preload(logger.transitions_directory)

    # Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = [2, 2] if not is_testing else [0, 0]
    explore_prob = 0.5 if not is_testing else 0.0

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action' : False,
                          'primitive_action' : None,
                          'best_pix_ind' : None,
                          'push_success' : False,
                          'grasp_success' : False,
                          'color_success' : False,
                          'place_success' : False,
                          'partial_stack_success': False,
                          'stack_height': 1,
                          'stack_rate': np.inf,
                          'trial_success_rate': np.inf,
                          'replay_iteration': 0,}
    best_stack_rate = np.inf

    # Choose the first color block to grasp, or None if not running in goal conditioned mode
    nonlocal_variables['stack'] = StackSequence(num_obj - num_extra_obj, grasp_color_task or place)
    if place:
        # If we are stacking we actually skip to the second block which needs to go on the first
        nonlocal_variables['stack'].next()

    def set_nonlocal_success_variables_false():
        nonlocal_variables['push_success'] = False
        nonlocal_variables['grasp_success'] = False
        nonlocal_variables['place_success'] = False
        # HK: Added color variable
        nonlocal_variables['grasp_color_success'] = False
        nonlocal_variables['place_color_success'] = False

    def check_stack_update_goal(place_check=False, top_idx=-1):
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
        if place_check:
            # Only reset while placing if the stack decreases in height!
            stack_shift = 1
        else:
            # only the place check expects the current goal to be met
            current_stack_goal = current_stack_goal[:-1]
            stack_shift = 0
        # TODO(ahundt) BUG Figure out why a real stack of size 2 or 3 and a push which touches no blocks does not pass the stack_check and ends up a MISMATCH in need of reset. (update: may now be fixed, double check then delete when confirmed)
        stack_matches_goal, nonlocal_variables['stack_height'] = robot.check_stack(current_stack_goal, top_idx=top_idx)
        nonlocal_variables['partial_stack_success'] = stack_matches_goal
        if nonlocal_variables['stack_height'] == 1:
            # A stack of size 1 does not meet the criteria for a partial stack success
            nonlocal_variables['partial_stack_success'] = False
            nonlocal_variables['stack_success'] = False

        max_workspace_height = len(current_stack_goal) - stack_shift
        # Has that stack gotten shorter than it was before? If so we need to reset
        needed_to_reset = nonlocal_variables['stack_height'] < max_workspace_height
        print('check_stack() stack_height: ' + str(nonlocal_variables['stack_height']) + ' stack matches current goal: ' + str(stack_matches_goal) + ' partial_stack_success: ' +
              str(nonlocal_variables['partial_stack_success']) + ' Does the code think a reset is needed: ' + str(needed_to_reset))
        # if place and needed_to_reset:
        # TODO(ahundt) BUG may reset push/grasp success too aggressively. If statement above and below for debugging, remove commented line after debugging complete
        if needed_to_reset:
            # we are two blocks off the goal, reset the scene.
            print('main.py check_stack() DETECTED A MISMATCH between the goal height: ' + str(max_workspace_height) +
                  ' and current workspace stack height: ' + str(nonlocal_variables['stack_height']) +
                  ', RESETTING the objects, goals, and action success to FALSE...')
            get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, trainer, '1')
            robot.reposition_objects()
            nonlocal_variables['stack'].reset_sequence()
            nonlocal_variables['stack'].next()
            # We needed to reset, so the stack must have been knocked over!
            # all rewards and success checks are False!
            set_nonlocal_success_variables_false()
        return needed_to_reset

    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def process_actions():
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
        successful_trial_count = 0
        trial_rate = np.inf
        while True:
            if nonlocal_variables['executing_action']:
                action_count += 1
                # Determine whether grasping or pushing should be executed based on network predictions
                best_push_conf = np.max(push_predictions)
                best_grasp_conf = np.max(grasp_predictions)
                if place:
                    best_place_conf = np.max(place_predictions)
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
                        push_frequency_one_in_n = 4
                        nonlocal_variables['primitive_action'] = 'push' if np.random.randint(0, push_frequency_one_in_n) == 0 else 'grasp'
                    else:
                        print('Strategy: exploit (exploration probability: %f)' % (explore_prob))
                trainer.is_exploit_log.append([0 if explore_actions else 1])
                logger.write_to_log('is-exploit', trainer.is_exploit_log)
                trainer.trial_log.append([nonlocal_variables['stack'].trial])
                logger.write_to_log('trial', trainer.trial_log)

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

                    # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
                    if nonlocal_variables['primitive_action'] == 'push':
                        nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
                        predicted_value = np.max(push_predictions)
                    elif nonlocal_variables['primitive_action'] == 'grasp':
                        nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
                        predicted_value = np.max(grasp_predictions)
                    elif nonlocal_variables['primitive_action'] == 'place':
                        nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(place_predictions), place_predictions.shape)
                        predicted_value = np.max(place_predictions)
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
                primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0], best_pix_y * heightmap_resolution + workspace_limits[1][0], valid_depth_heightmap[best_pix_y][best_pix_x] + workspace_limits[2][0]]

                # If pushing, adjust start position, and make sure z value is safe and not too low
                if nonlocal_variables['primitive_action'] == 'push': # or nonlocal_variables['primitive_action'] == 'place':
                    finger_width = 0.02
                    safe_kernel_width = int(np.round((finger_width/2)/heightmap_resolution))
                    local_region = valid_depth_heightmap[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[0]), max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1])]
                    if local_region.size == 0:
                        safe_z_position = workspace_limits[2][0]
                    else:
                        safe_z_position = np.max(local_region) + workspace_limits[2][0]
                    primitive_position[2] = safe_z_position

                # Save executed primitive where [0, 1, 2] corresponds to [push, grasp, place]
                if nonlocal_variables['primitive_action'] == 'push':
                    trainer.executed_action_log.append([0, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]])  # 0 - push
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    trainer.executed_action_log.append([1, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]])  # 1 - grasp
                elif nonlocal_variables['primitive_action'] == 'place':
                    trainer.executed_action_log.append([2, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][2]])  # 2 - place
                logger.write_to_log('executed-action', trainer.executed_action_log)

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    push_pred_vis = trainer.get_prediction_vis(push_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, push_pred_vis, 'push')
                    cv2.imwrite('visualization.push.png', push_pred_vis)
                    grasp_pred_vis = trainer.get_prediction_vis(grasp_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, grasp_pred_vis, 'grasp')
                    cv2.imwrite('visualization.grasp.png', grasp_pred_vis)
                    if place:
                        place_pred_vis = trainer.get_prediction_vis(place_predictions, color_heightmap, nonlocal_variables['best_pix_ind'])
                        logger.save_visualizations(trainer.iteration, place_pred_vis, 'place')
                        cv2.imwrite('visualization.place.png', place_pred_vis)

                # Initialize variables that influence reward
                set_nonlocal_success_variables_false()
                change_detected = False
                if place:
                    current_stack_goal = nonlocal_variables['stack'].current_sequence_progress()

                # Execute primitive
                if nonlocal_variables['primitive_action'] == 'push':
                    nonlocal_variables['push_success'] = robot.push(primitive_position, best_rotation_angle, workspace_limits)
                    if place:
                        # Check if the push caused a topple, size shift zero because
                        # place operations expect increased height,
                        # while push expects constant height.
                        needed_to_reset = check_stack_update_goal()
                    if not place or not needed_to_reset:
                        print('Push motion successful (no crash, need not move blocks): %r' % (nonlocal_variables['push_success']))
                elif nonlocal_variables['primitive_action'] == 'grasp':
                    grasp_count += 1
                    # TODO(ahundt) this probably will cause threading conflicts, add a mutex
                    if nonlocal_variables['stack'].object_color_index is not None and grasp_color_task:
                        grasp_color_name = robot.color_names[int(nonlocal_variables['stack'].object_color_index)]
                        print('Attempt to grasp color: ' + grasp_color_name)
                    nonlocal_variables['grasp_success'], nonlocal_variables['grasp_color_success'] = robot.grasp(primitive_position, best_rotation_angle, object_color=nonlocal_variables['stack'].object_color_index)
                    print('Grasp successful: %r' % (nonlocal_variables['grasp_success']))
                    if place:
                        # when we are stacking we must also check the stack in case we caused it to topple
                        top_idx = -1
                        if nonlocal_variables['grasp_success']:
                            # we will need to check the second from top block for the stack
                            top_idx = -2
                        # check if a failed grasp led to a topple, or if the top block was grasped
                        # TODO(ahundt) in check_stack() support the check after a specific grasp in case of successful grasp topple. Perhaps allow the top block to be specified?
                        needed_to_reset = check_stack_update_goal(top_idx=top_idx)
                    if nonlocal_variables['grasp_success']:
                        # robot.restart_sim()
                        successful_grasp_count += 1
                        if grasp_color_task:
                            if nonlocal_variables['grasp_color_success']:
                                successful_color_grasp_count += 1
                            if not place:
                                # reposition the objects if we aren't also attempting to place correctly.
                                robot.reposition_objects()

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
                    needed_to_reset = check_stack_update_goal(place_check=True)
                    if not needed_to_reset and nonlocal_variables['place_success'] and nonlocal_variables['partial_stack_success']:
                        partial_stack_count += 1
                        nonlocal_variables['stack'].next()
                        next_stack_goal = nonlocal_variables['stack'].current_sequence_progress()
                        if len(next_stack_goal) < len(current_stack_goal):
                            nonlocal_variables['stack_success'] = True
                            stack_count += 1
                            # full stack complete! reset the scene
                            successful_trial_count += 1
                            get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, trainer, '1')
                            robot.reposition_objects()
                            nonlocal_variables['stack'].reset_sequence()
                            nonlocal_variables['stack'].next()
                    # TODO(ahundt) perhaps reposition objects every time a partial stack step fails (partial_stack_success == false) to avoid weird states?

                trainer.grasp_success_log.append([int(nonlocal_variables['grasp_success'])])
                if grasp_color_task:
                    trainer.color_success_log.append([int(nonlocal_variables['color_success'])])
                if place:
                    # place trainer logs are updated in process_actions()
                    trainer.stack_height_log.append([int(nonlocal_variables['stack_height'])])
                    trainer.partial_stack_success_log.append([int(nonlocal_variables['partial_stack_success'])])
                    trainer.place_success_log.append([int(nonlocal_variables['place_success'])])
                    
                    if partial_stack_count > 0:
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
                          '  stack_successes: ' + str(stack_count) + ' trial_success_rate: ' + str(trial_rate) + ' stack goal: ' + str(current_stack_goal))

                nonlocal_variables['executing_action'] = False
            # TODO(ahundt) this should really be using proper threading and locking algorithms
            time.sleep(0.01)

    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    prev_primitive_action = None
    prev_reward_value = None

    # Start main training/testing loop, max_iter == 0 or -1 goes forever.
    while max_iter < 0 or trainer.iteration < max_iter:
        print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim:
            robot.check_sim()

        # Get latest RGB-D image
        valid_depth_heightmap, color_heightmap, depth_heightmap, color_img, depth_img = get_and_save_images(
            robot, workspace_limits, heightmap_resolution, logger, trainer)

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.02] = 1
        empty_threshold = 300
        if is_sim and is_testing:
            empty_threshold = 10
        if np.sum(stuff_count) < empty_threshold or (is_sim and no_change_count[0] + no_change_count[1] > 10):
            if is_sim:
                print('There have not been changes to the objects for for a long time [push, grasp]: ' + str(no_change_count) +
                      ', or there are not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
                robot.restart_sim()
                robot.add_objects()
                if is_testing: # If at end of test run, re-load original weights (before test run)
                    trainer.model.load_state_dict(torch.load(snapshot_file))
                if place:
                    set_nonlocal_success_variables_false()
                    nonlocal_variables['stack'].reset_sequence()
                    nonlocal_variables['stack'].next()
            else:
                # print('Not enough stuff on the table (value: %d)! Pausing for 30 seconds.' % (np.sum(stuff_count)))
                # time.sleep(30)
                print('Not enough stuff on the table (value: %d)! Flipping over bin of objects...' % (np.sum(stuff_count)))
                robot.restart_real()

            no_change_count = [0, 0]
            trainer.clearance_log.append([trainer.iteration])
            logger.write_to_log('clearance', trainer.clearance_log)
            if is_testing and not place and len(trainer.clearance_log) >= max_test_trials:
                exit_called = True # Exit after training thread (backprop and saving labels)

            continue

        if is_testing and place and nonlocal_variables['stack'].trial > max_test_trials:
            exit_called = True

        if not exit_called:

            # Run forward pass with network to get affordances
            if nonlocal_variables['stack'].is_goal_conditioned_task and grasp_color_task:
                goal_condition = np.array([nonlocal_variables['stack'].current_one_hot()])
            else:
                goal_condition = None

            push_predictions, grasp_predictions, place_predictions, state_feat = trainer.forward(
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
            trainer.label_value_log.append([label_value])
            # label-value is also known as expected_reward in trainer.get_label_value()
            logger.write_to_log('label-value', trainer.label_value_log)
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

            # Backpropagate
            trainer.backprop(prev_color_heightmap, prev_valid_depth_heightmap, prev_primitive_action, prev_best_pix_ind, label_value, goal_condition=prev_goal_condition)

            # Adjust exploration probability
            if not is_testing:
                explore_prob = max(0.5 * np.power(0.9998, trainer.iteration), 0.1) if explore_rate_decay else 0.5

            # Do sampling for experience replay
            if experience_replay_enabled and prev_reward_value is not None and not is_testing:
                # Here we will try to sample a reward value from the same action as the current one
                # which differs from the most recent reward value to reduce the chance of catastrophic forgetting.
                # TODO(ahundt) experience replay is very hard-coded with lots of bugs, won't evaluate all reward possibilities, and doesn't deal with long range time dependencies.
                experience_replay(method, prev_primitive_action, prev_reward_value, trainer, grasp_color_task, logger, nonlocal_variables, place, goal_condition)

            # Save model snapshot
            if not is_testing:
                logger.save_backup_model(trainer.model, method)
                if trainer.iteration % 50 == 0:
                    logger.save_model(trainer.model, method)
                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()
                # Save model if we are at a new best stack rate
                if place and trainer.iteration >= 1000 and nonlocal_variables['stack_rate'] < best_stack_rate:
                    best_stack_rate = nonlocal_variables['stack_rate']
                    stack_rate_str = method + '-best-stack-rate'
                    logger.save_backup_model(trainer.model, stack_rate_str)
                    logger.save_model(trainer.model, stack_rate_str)
                    if trainer.use_cuda:
                        trainer.model = trainer.model.cuda()

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            if experience_replay_enabled and prev_reward_value is not None and not is_testing:
                # do some experience replay while waiting, rather than sleeping
                experience_replay(method, prev_primitive_action, prev_reward_value, trainer, grasp_color_task, logger, nonlocal_variables, place, goal_condition)
            else:
                time.sleep(0.01)
            time_elapsed = time.time()-iteration_time_0
            if int(time_elapsed) > 20:
                # TODO(ahundt) double check that this doesn't screw up state completely for future trials...
                print('ERROR: PROBLEM DETECTED IN SCENE, NO CHANGES FOR OVER 20 SECONDS, RESETTING THE OBJECTS TO RECOVER...')
                get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, trainer, '1')
                robot.reposition_objects()

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
        prev_push_predictions = push_predictions.copy()
        prev_grasp_predictions = grasp_predictions.copy()
        prev_place_predictions = place_predictions
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']
        # TODO(ahundt) BUG We almost certainly need to copy nonlocal_variables['stack']
        prev_stack = nonlocal_variables['stack']
        prev_goal_condition = goal_condition
        # HK: check color_success arguments
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
        # HK: TODO

        print('Trainer iteration: %f' % (trainer.iteration))

def get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, trainer, filename_poststring='0'):
    # Get latest RGB-D image
    color_img, depth_img = robot.get_camera_data()
    depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration
    #print(color_img)
    # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
    color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

    # Save RGB-D images and RGB-D heightmaps
    logger.save_images(trainer.iteration, color_img, depth_img, filename_poststring)
    logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, filename_poststring)
    return valid_depth_heightmap, color_heightmap, depth_heightmap, color_img, depth_img

def experience_replay(method, prev_primitive_action, prev_reward_value, trainer, grasp_color_task, logger, nonlocal_variables, place, goal_condition):
    # Here we will try to sample a reward value from the same action as the current one
    # which differs from the most recent reward value to reduce the chance of catastrophic forgetting.
    # TODO(ahundt) experience replay is very hard-coded with lots of bugs, won't evaluate all reward possibilities, and doesn't deal with long range time dependencies.
    sample_primitive_action = prev_primitive_action
    sample_primitive_action_id = ACTION_TO_ID[sample_primitive_action]
    # executed_action_log includes the action, push grasp or place, and the best pixel index
    actions = np.asarray(trainer.executed_action_log)[1:trainer.iteration,0]
    prev_success = np.array(bool(prev_reward_value))

    # Get samples of the same primitive but with different success results
    if sample_primitive_action == 'push':
        # sample_primitive_action_id = 0
        sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.change_detected_log)[1:trainer.iteration,0] != prev_success, 
                                                actions == sample_primitive_action_id))
    elif sample_primitive_action == 'grasp':
        # sample_primitive_action_id = 1
        sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.grasp_success_log)[1:trainer.iteration,0] != prev_success, 
                                                actions == sample_primitive_action_id))
    elif sample_primitive_action == 'place':
        sample_ind = np.argwhere(np.logical_and(np.asarray(trainer.partial_stack_success_log)[1:trainer.iteration,0] != prev_success, 
                                                actions == sample_primitive_action_id))
    else:
        raise NotImplementedError('ERROR: ' + sample_primitive_action + ' action is not yet supported in experience replay')

    if sample_ind.size == 0 and prev_reward_value is not None and trainer.iteration > 2:
        print('Experience Replay: We do not have samples for the ' + sample_primitive_action + ' action with a success state of ' + str(not prev_success) + ', so sampling from the whole history.')
        sample_ind = np.arange(1,trainer.iteration-1).reshape(trainer.iteration-2, 1)

    if sample_ind.size > 0:
        # Find sample with highest surprise value
        if method == 'reactive':
            sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:,0]] - (1 - sample_reward_value))
        elif method == 'reinforcement':
            sample_surprise_values = np.abs(np.asarray(trainer.predicted_value_log)[sample_ind[:,0]] - np.asarray(trainer.label_value_log)[sample_ind[:,0]])
        sorted_surprise_ind = np.argsort(sample_surprise_values[:,0])
        sorted_sample_ind = sample_ind[sorted_surprise_ind, 0]
        pow_law_exp = 2
        rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(sample_ind.size-1)))
        # sample_iteration is the actual time step on which we will run experience replay
        sample_iteration = sorted_sample_ind[rand_sample_ind]
        sample_primitive_action_id = trainer.executed_action_log[sample_iteration][0]
        sample_primitive_action = ID_TO_ACTION[sample_primitive_action_id]
        sample_reward_value = trainer.reward_value_log[sample_iteration]
        nonlocal_variables['replay_iteration'] += 1
        print('Experience replay %d: history timestep index %d, action: %s, surprise value: %f' % (nonlocal_variables['replay_iteration'], sample_iteration, str(sample_primitive_action), sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

        # Load sample RGB-D heightmap
        sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration)))
        sample_color_heightmap = cv2.cvtColor(sample_color_heightmap, cv2.COLOR_BGR2RGB)
        sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration)), -1)
        sample_depth_heightmap = sample_depth_heightmap.astype(np.float32)/100000

        # Compute forward pass with sample
        if nonlocal_variables['stack'].is_goal_conditioned_task and grasp_color_task:
            exp_goal_condition = [trainer.goal_condition_log[sample_iteration]]
            next_goal_condition = [trainer.goal_condition_log[sample_iteration+1]]
        else:
            exp_goal_condition = None
            next_goal_condition = None
            sample_color_success = None

        if place:
            # print('place loading stack_height_log sample_iteration: ' + str(sample_iteration) + ' log len: ' + str(len(trainer.stack_height_log)))
            sample_stack_height = int(trainer.stack_height_log[sample_iteration][0])
            next_stack_height = int(trainer.stack_height_log[sample_iteration+1][0])
        else:
            # set to 1 because stack height is used as the reward multiplier
            sample_stack_height = 1
            next_stack_height = 1

        sample_push_predictions, sample_grasp_predictions, sample_place_predictions, sample_state_feat = trainer.forward(
            sample_color_heightmap, sample_depth_heightmap, is_volatile=True, goal_condition=exp_goal_condition)

        # Load next sample RGB-D heightmap
        next_sample_color_heightmap = cv2.imread(os.path.join(logger.color_heightmaps_directory, '%06d.0.color.png' % (sample_iteration+1)))
        next_sample_color_heightmap = cv2.cvtColor(next_sample_color_heightmap, cv2.COLOR_BGR2RGB)
        next_sample_depth_heightmap = cv2.imread(os.path.join(logger.depth_heightmaps_directory, '%06d.0.depth.png' % (sample_iteration+1)), -1)
        next_sample_depth_heightmap = next_sample_depth_heightmap.astype(np.float32)/100000
        # TODO(ahundt) TODO(hkwon14) Fix success checks to be performed correctly for all mode combinations of grasping, color grasping, and stacking, rewards must match get_label_value() in trainer.py, prefereably in an easy to use way.
        # TODO(ahundt) TODO(hkwon14) tune sample_reward_value?
        sample_place_success = None
        # note that push success is always true in robot.push, and didn't affect get_label_value at the time of writing.
        sample_push_success = True
        sample_change_detected = trainer.change_detected_log[sample_iteration]
        sample_grasp_success = trainer.grasp_success_log[sample_iteration]
        if place:
            sample_place_success = trainer.partial_stack_success_log[sample_iteration]
        # in this case grasp_color_task is True
        if exp_goal_condition is not None:
            sample_color_success = trainer.color_success_log[sample_iteration]

        # if no_height_reward:  # TODO(ahundt) why does the args.no_height_reward line below work and the regular no_height_reward here broken?
        if args.no_height_reward:
            # used to assess the value of the reward multiplier
            reward_multiplier = 1
        else:
            reward_multiplier = sample_stack_height
        # TODO(hkwon14) This mix of current and next parameters (like next_sample_color_heightmap and sample_push_success) seems a likely spot for a bug, we must make sure we haven't broken the behavior. ahundt has already fixed one bug here.
        # get_label_value does the forward pass for us to backprop, even if we don't use the return values.
        new_sample_label_value, _ = trainer.get_label_value(
            sample_primitive_action, sample_push_success, sample_grasp_success, sample_change_detected,
            sample_push_predictions, sample_grasp_predictions, next_sample_color_heightmap, next_sample_depth_heightmap,
            sample_color_success, goal_condition=exp_goal_condition, prev_place_predictions=sample_place_predictions,
            place_success=sample_place_success, reward_multiplier=reward_multiplier)

        # Get labels for sample and backpropagate
        sample_best_pix_ind = (np.asarray(trainer.executed_action_log)[sample_iteration, 1:4]).astype(int)
        trainer.backprop(sample_color_heightmap, sample_depth_heightmap, sample_primitive_action, sample_best_pix_ind,
                         trainer.label_value_log[sample_iteration], goal_condition=exp_goal_condition)

        # Recompute prediction value and label for replay buffer
        if sample_primitive_action == 'push':
            trainer.predicted_value_log[sample_iteration] = [np.max(sample_push_predictions)]
            # trainer.label_value_log[sample_iteration] = [new_sample_label_value]
        elif sample_primitive_action == 'grasp':
            trainer.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]
        elif sample_primitive_action == 'place':
            trainer.predicted_value_log[sample_iteration] = [np.max(sample_place_predictions)]
            # trainer.label_value_log[sample_iteration] = [new_sample_label_value]

    else:
        print('Experience Replay: 0 prior training samples. Skipping experience replay.')


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                    help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,                                help='number of objects to add to simulation')
    parser.add_argument('--num_extra_obj', dest='num_extra_obj', type=int, action='store', default=0,                     help='number of secondary objects, like distractors, to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,                           help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,                           help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--method', dest='method', action='store', default='reinforcement',                               help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
    parser.add_argument('--push_rewards', dest='push_rewards', action='store_true', default=False,                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,              help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,          help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    parser.add_argument('--grasp_only', dest='grasp_only', action='store_true', default=False)
    parser.add_argument('--random_weights', dest='random_weights', action='store_true', default=False,                    help='use random weights rather than weights pretrained on ImageNet')
    parser.add_argument('--max_iter', dest='max_iter', action='store', type=int, default=-1,                              help='max iter for training. -1 (default) trains indefinitely.')


    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=100,                help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')
    parser.add_argument('--place', dest='place', action='store_true', default=False,                                      help='enable placing of objects')
    parser.add_argument('--no_height_reward', dest='no_height_reward', action='store_true', default=False,                                      help='disable stack height reward multiplier')
    parser.add_argument('--grasp_color_task', dest='grasp_color_task', action='store_true', default=False,              help='enable grasping specific colored objects')
    parser.add_argument('--grasp_count', dest='grasp_cout', type=int, action='store', default=0,                                help='number of successful task based grasps')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
