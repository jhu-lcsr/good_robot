import time
import os
import random
import threading
import argparse
from logger import Logger
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
from robot import Robot
import utils
from utils import StackSequence, annotate_success_manually
from main import get_and_save_images
from annotate_data import Pair 

logger = Logger(True, "/Users/Elias/scratch", args=None, dir_name="/Users/Elias/scratch")
############### Testing Block Stacking #######
is_sim = True# Run in simulation?
obj_mesh_dir = os.path.abspath('objects/blocks') if is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
num_obj = 4 if is_sim else None # Number of objects to add to simulation
tcp_host_ip = args.tcp_host_ip if not is_sim else None # IP and port to robot arm as TCP client (UR5)
tcp_port = args.tcp_port if not is_sim else None
rtc_host_ip = args.rtc_host_ip if not is_sim else None # IP and port to robot arm as real-time client (UR5)
rtc_port = args.rtc_port if not is_sim else None
if is_sim:
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
else:
    workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
heightmap_resolution = 0.002 # Meters per pixel of heightmap
random_seed = 1234
force_cpu = False

# -------------- Testing options --------------
is_testing = False
max_test_trials = 1 # Maximum number of test runs per case/scenario
test_preset_cases = False
test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

# ------ Pre-loading and logging options ------
#load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
#snapshot_file = os.path.abspath(args.snapshot_file)  if load_snapshot else None
#continue_logging = args.continue_logging # Continue logging from previous session
#logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
#save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True


# Set random seed
np.random.seed(random_seed)
# Do we care about color? Switch to
# True to run a color order stacking test,
# False tests stacking order does not matter.
grasp_color_task = False
# are we doing a stack even if we don't care about colors
place_task = True
# place in rows instead of stacks
check_row = False
# if placing in rows, how far apart to set the blocks?
separation = 0.055
distance_threshold = 0.08

robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
              tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
              is_testing, test_preset_cases, test_preset_file,
              place=place_task, grasp_color_task=grasp_color_task)
stacksequence = StackSequence(num_obj, is_goal_conditioned_task=grasp_color_task or place_task)

print('full stack sequence: ' + str(stacksequence.object_color_sequence))
best_rotation_angle = 3.14
blocks_to_move = num_obj - 1
num_stacks = 16
original_position = np.array([-0.6, 0.25, 0])
test_failure = False

valid_depth_heightmap, color_heightmap, depth_heightmap, color_img, depth_img = get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, None, depth_channels_history=False, save_image=False)

prev_heightmap = color_heightmap 
next_heightmap = None

for stack in range(num_stacks):
    print('++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++++++ Making New Stack                  ++++++++')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++')

    theta = 2 * np.pi * stack / num_stacks
    
    # move the first block in order to have it in a standard position.
    print('orienting first block')
    stack_goal = stacksequence.current_sequence_progress()
    block_to_move = stack_goal[0]
    block_positions, block_orientations = robot.get_obj_positions_and_orientations()
    primitive_position = block_positions[block_to_move]
    rotation_angle = block_orientations[block_to_move][2]
    robot.grasp(primitive_position, rotation_angle,
                object_color=block_to_move)
    block_positions = robot.get_obj_positions_and_orientations()[0]
    # creates the ideal stack by fixing rotation angle
    place = robot.place(original_position.copy(), theta + np.pi / 2)
    print('place initial: ' + str(place))
    
    for i in range(blocks_to_move):
        print('----------------------------------------------')
        stacksequence.next()
        stack_goal = stacksequence.current_sequence_progress()

        pair = Pair.from_nonsim_main_idxs(color_heightmap,
                                   valid_depth_heightmap,
                                   is_row = check_row)

        block_to_move = stack_goal[-1]
        print('move block: ' + str(i) + ' current stack goal: ' + str(stack_goal))
        block_positions, block_orientations = robot.get_obj_positions_and_orientations()

        
        valid_depth_heightmap, color_heightmap, depth_heightmap, color_img, depth_img = get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, None, depth_channels_history=False, save_image=False)
        prev_heightmap = color_heightmap

        primitive_position = block_positions[block_to_move]
        rotation_angle = block_orientations[block_to_move][2]
        robot.grasp(primitive_position, rotation_angle, object_color=block_to_move)
        base_block_to_place = stack_goal[0]
        if check_row:
            primitive_position = original_position + (i + 1) * separation * np.array([np.cos(theta), np.sin(theta), 0])
        else:
            block_positions = robot.get_obj_positions_and_orientations()[0]
            primitive_position = block_positions[base_block_to_place]

            # place height should be on the top of the stack. otherwise the sim freaks out
            stack_z_height = 0
            for block_pos in block_positions:
                if block_pos[2] > stack_z_height and block_pos[2] < workspace_limits[2][1]:
                    stack_z_height = block_pos[2]
            primitive_position[2] = stack_z_height

        place = robot.place(primitive_position, theta + np.pi / 2)
        print('place ' + str(i) + ' : ' + str(place))

        valid_depth_heightmap, color_heightmap, depth_heightmap, color_img, depth_img = get_and_save_images(robot, workspace_limits, heightmap_resolution, logger, None, depth_channels_history=False, save_image=False)
        next_heightmap = color_heightmap

        # check if we don't care about color
        if not grasp_color_task:
            # Deliberately change the goal stack order to test the non-ordered check
            stack_goal = np.random.permutation(stack_goal)
            print('fake stack goal to test any stack order: ' + str(stack_goal))
        if check_row:
            stack_success, height_count = robot.check_row(stack_goal, distance_threshold=distance_threshold)
        else:
            success_code, comment = annotate_success_manually("command", prev_heightmap, next_heightmap)
            stack_success, height_count = robot.check_stack(stack_goal, distance_threshold=distance_threshold)
            print('stack success part ' + str(i+1) + ' of ' + str(blocks_to_move) + ': ' + str(stack_success))
    # reset scene
    robot.reposition_objects()
    # determine first block to grasp
    stacksequence.next()
