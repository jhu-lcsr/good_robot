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
              is_testing, test_preset_cases, test_preset_file)
block = robot.get_obj_positions_and_orientations()
primitive_position = block[0][0]
print(primitive_position)
best_rotation_angle = 3.14
print(best_rotation_angle)
primitive_position1 = block[0][1]

primitive_position2 = block[0][2]
primitive_position3 = block[0][3]

robot.grasp(primitive_position, best_rotation_angle, workspace_limits)
robot.place(primitive_position1, best_rotation_angle, workspace_limits)
robot.grasp(primitive_position2, best_rotation_angle, workspace_limits)
block1 = robot.get_obj_positions_and_orientations()
robot.place(block1[0][0], best_rotation_angle, workspace_limits)

robot.grasp(primitive_position3, best_rotation_angle, workspace_limits)
robot.place(block1[0][0], best_rotation_angle, workspace_limits)



