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
from main import StackSequence


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
# are we stacking in Z or some other dimension?
stack_axis = 1
# if placing in rows, how far apart to set the blocks?
separation = 0.06

robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
              tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
              is_testing, test_preset_cases, test_preset_file,
              place=place_task, grasp_color_task=grasp_color_task)
stacksequence = StackSequence(num_obj, is_goal_conditioned_task=grasp_color_task or place_task)

print('full stack sequence: ' + str(stacksequence.object_color_sequence))
best_rotation_angle = 3.14
blocks_to_move = num_obj - 1
num_stacks = 10

for stack in range(num_stacks):
    print('++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++++++ Making New Stack                  ++++++++')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++')
    for i in range(blocks_to_move):
        print('----------------------------------------------')
        stacksequence.next()
        stack_goal = stacksequence.current_sequence_progress()
        block_to_move = stack_goal[-1]
        print('move block: ' + str(i) + ' current stack goal: ' + str(stack_goal))
        block_positions = robot.get_obj_positions_and_orientations()[0]
        primitive_position = block_positions[block_to_move]
        robot.grasp(primitive_position, best_rotation_angle, object_color=block_to_move)
        block_positions = robot.get_obj_positions_and_orientations()[0]
        base_block_to_place = stack_goal[-2]
        primitive_position = block_positions[base_block_to_place]
        print('primitive_position', primitive_position)
        if stack_axis != 2:
            primitive_position[stack_axis] += separation
        place = robot.place(primitive_position, best_rotation_angle)
        print('place ' + str(i) + ' : ' + str(place))
        # check if we don't care about color
        if not grasp_color_task:
            # Deliberately change the goal stack order to test the non-ordered check
            stack_goal = np.random.permutation(stack_goal)
            print('fake stack goal to test any stack order: ' + str(stack_goal))
        stack_success, height_count = robot.check_stack(stack_goal, stack_axis=stack_axis)
        print('stack success part ' + str(i+1) + ' of ' + str(blocks_to_move) + ': ' + str(stack_success))
    # reset scene
    robot.reposition_objects()
    # determine first block to grasp
    stacksequence.next()
