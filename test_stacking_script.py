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

############### Testing Block Stacking #######

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

    def current_one_hot(self):
        """ Return the one hot encoding for the current specific object.
        """
        return self.object_color_one_hot_encodings[self.object_color_index]

    def sequence_one_hot(self):
        """ Return the one hot encoding for the entire stack sequence.
        """
        return np.concatenate(object_color_one_hot_encodings)

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

    def object_color_sequence_func(self):
        return self.object_color_sequence

    def next(self):
        self.total_steps += 1
        if self.is_goal_conditioned_task:
            self.object_color_index += 1
            if not self.object_color_index < self.num_obj:
                self.reset_sequence()
######################

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

robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
              tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
              is_testing, test_preset_cases, test_preset_file,
              place=True, grasp_color_task=True)
stacksequence = StackSequence(num_obj, is_goal_conditioned_task=True)

block = robot.get_obj_positions_and_orientations()
primitive_position = block[0][0]
best_rotation_angle = 3.14

primitive_position1 = block[0][1]
primitive_position2 = block[0][2]
primitive_position3 = block[0][3]

robot.grasp(primitive_position, best_rotation_angle)
place0 = robot.place(primitive_position1, best_rotation_angle)
block1 = robot.get_obj_positions_and_orientations()
print('place0 '+ str(place0))

robot.grasp(primitive_position2, best_rotation_angle)
block1 = robot.get_obj_positions_and_orientations()
place1 = robot.place(block1[0][0], best_rotation_angle)
print('place1 '+ str(place1))

robot.grasp(primitive_position3, best_rotation_angle)
place2 = robot.place(block1[0][0], best_rotation_angle)
print('place2 '+ str(place2))


#stack_goal = stacksequence.object_color_sequence_func()
stack_goal = [3,0,2]
print('stack_goal: ' + str(stack_goal))

stack_success = robot.check_stack(stack_goal)
print('stack '+ str(stack_success))

