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
import csv
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
try:
    import efficientnet_pytorch
    from efficientnet_pytorch import EfficientNet
except ImportError:
    print('efficientnet_pytorch is not available, using densenet. '
          'Try installing https://github.com/ahundt/EfficientNet-PyTorch for all features.'
          'A version of EfficientNets without dilation can be installed with the command:'
          '    pip3 install efficientnet-pytorch --user --upgrade'
          'See https://github.com/lukemelas/EfficientNet-PyTorch for details')
    efficientnet_pytorch = None

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

#####################################################################
# Continue logging from previous session
continue_logging = False # Set True if continue logging from previous session
split = 'train' # Set ['train', 'val', 'test']
root = './logs_for_classifier'

logging_directory = os.path.join(root, split)
if split == 'train':
    num_stacks = 33400
    max_iterations = 100000
elif split == 'test' or split == 'val':
    num_stacks = 3400
    max_iterations = 10000

labels = []
data_num = 1
#####################################################################

# ------ Pre-loading and logging options ------
# load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
# snapshot_file = os.path.abspath(args.snapshot_file)  if load_snapshot else None
# continue_logging = False # Continue logging from previous session
#logging_directory = './logs_for_classifier/training'
logging_directory = os.path.abspath(logging_directory) if continue_logging else os.path.abspath('logs_for_classifier')
# save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True

# Initialize data logger
logger = Logger(continue_logging, logging_directory)
# logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
# logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters

# Set random seed
np.random.seed(random_seed)
# Do we care about color? Switch to
# True to run a color order stacking test,
# False tests stacking order does not matter.
grasp_color_task = False
# are we doing a stack even if we don't care about colors
place_task = True

robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
              tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
              is_testing, test_preset_cases, test_preset_file,
              place=place_task, grasp_color_task=grasp_color_task)
stacksequence = StackSequence(num_obj, is_goal_conditioned_task=grasp_color_task or place_task)

print('full stack sequence: ' + str(stacksequence.object_color_sequence))
best_rotation_angle = 3.14
blocks_to_move = num_obj - 1

############## Load Image Classifier Weights ###############
num_class = 4
checkpoint_path = "./eval-20190818-154803-6ebd1fa-stack_height-efficientnet-0/model_best.pth.tar"
height_count_sum = 0
stack_success_sum = 0

model_stack = EfficientNet.from_name('efficientnet-b0')
#model = nn.DataParallel(model)
#model_stack = model_stack.cuda()
checkpoint = torch.load(checkpoint_path)
model_stack.load_state_dict(checkpoint['state_dict'])
model_stack.eval()

if continue_logging  == False:
    iteration = 0
else:
    label_text = os.path.join(logging_directory, 'data','color-images', 'stack_label.txt')
    myfile = open(label_text, 'r')
    myfiles = [line.split(' ') for line in myfile.readlines()]
    iteration = int(myfiles[-1][1]) +1

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
        place = robot.place(primitive_position, best_rotation_angle)
        print('place ' + str(i) + ' : ' + str(place))
        # check if we don't care about color
        if not grasp_color_task:
            # Deliberately change the goal stack order to test the non-ordered check
            stack_goal = np.random.permutation(stack_goal)
            print('fake stack goal to test any stack order: ' + str(stack_goal))
        stack_success, height_count = robot.check_stack(stack_goal)
        #######################################
        stack_class = height_count - 1
        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # Save RGB-D images and RGB-D heightmaps
        #logger.save_images(iteration, color_img, depth_img, stack_class) # Used stack_class instead of mode
        #logger.save_heightmaps(iteration, color_heightmap, valid_depth_heightmap, stack_class) # Used stack_class instead of mode
        ###########################################

        stack_success_classifier, height_count_classifier= robot.stack_reward(model_stack, depth_heightmap, stack_goal)
        filename = '%06d.%s.color.png' % (iteration, stack_class)
        if continue_logging:
            with open(label_text,"a") as f:
                f.writelines("\n")
                name = [filename,' ', str(iteration),' ', str(stack_class)]
                f.writelines(name)
                f.close()
        else:
            labels.append([filename,iteration,stack_class])
            logger.save_label('stack_label', labels)
        print('stack success part ' + str(i+1) + ' of ' + str(blocks_to_move) + ': ' + str(stack_success) +  ':' + str(height_count) +':' + str(stack_class))
        print('stack success classifier part ' + str(i+1) + ' of ' + str(blocks_to_move) + ': ' + str(stack_success_classifier) +  ':' + str(height_count) +':' + str(stack_class_classifier))
        iteration += 1
        if height_count_classifier == height_count:
            height_count_sum += 1
        if stack_success_classifier== stack_success:
            stack_success_sum += 1
        print('stack height classifier accuracy ' + str(i+1) + ' height_count ' + str(height_count_sum/iteration))
        print('stack success classifier accuracy ' + str(i+1) + ' stack_success ' + str(stack_success_sum/iteration))

    # reset scene
    robot.reposition_objects()
    # determine first block to grasp
    stacksequence.next()
    if iteration > max_iterations:
        break
