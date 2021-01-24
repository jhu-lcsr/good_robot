#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  generate_logoblocks_images.py
#  
#  Copyright 2020 Zhuohong He <zooey@zooey-HP-ENVY-Laptop-13-ad1xx>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import os
import math
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm 
import cv2
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation as R

from robot import Robot
from data import DatasetReader, SimpleTrajectory


class BlockSetter(object):
    """ Class to manipulate logo blocks and capture images used for training. """

    def __init__(self,
                 pos: List[Tuple[float]], 
                 rot: List[Tuple[float]], 
                 blocks_path: str,
                 num_obj: int,
                 shift: List[float]):
        """
        Load an initial setup, returns a new instance of the Robot 
        object. If robot class is already initialized, run "load_setup"
        instead.
        """
        self.num_obj = num_obj
        self.shift = shift
        
        # Setting up the Robot object
        is_sim = True
        self.obj_mesh_dir = os.path.abspath(blocks_path)
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.5]])
        is_testing = True
        max_test_trials = 1 # Maximum number of test runs per case/scenario
        test_preset_cases = False
        test_preset_file = None
        
        # Creating a list for cube position and orientation
        test_preset_arr = []
        # Reordering JSON values (y axis is up in sim, z axis is up in JSON)
        self.ordering = [2,0,1]
        # 5cm Blocks
        self.side_len = 0.035
        self.scale = self.side_len*0.001/0.15
        # 14x14 grid
        self.grid_dim = 14
        self.grid_len = self.grid_dim*self.side_len
        
        # Create array to pass initial position of blocks
        for i in range(len(pos[0])):
            # Convert values from dataset to match vrep axes
            state = [(pos[0][i][j] + self.shift[k]) * (self.grid_len/2) for k, j in enumerate(self.ordering)]
            state[1] = -state[1]
            orient = rot[0][i]
            
            # Convert quaternion to euler angles
            if len(orient)==4:
                orient = self.quat2euler(orient)
            
            test_preset_arr.append([state, orient])
        
        # Move unused objects out of the camera frame
        for i in range(self.num_obj-len(pos[0])):
            out_state, out_ori = [-0.75 - i*(self.side_len+0.01), 0, self.side_len], [0, 0, 0]
            test_preset_arr.append([out_state, out_ori])
        
        tcp_host_ip, tcp_port, rtc_host_ip, rtc_port = None, None, None, None
        
        self.robot = Robot(is_sim, self.obj_mesh_dir, self.num_obj, workspace_limits,
                  tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                  is_testing, test_preset_cases, test_preset_file, test_preset_arr, 
                  capture_logoblock_dataset=True, obj_scale=self.scale, textured=True)


    def quat2euler(self, quatArr):
        """ Returns euler angles (list) corresponding to a quaternion input (list) """
        # Play around with euler angle conventions to fix texture orientation.
        temp = R.from_quat(quatArr).as_euler('zxy', degrees=False)
        return [r.item() for r in temp]


    def load_setup(self, positions, orientations, prevPositions, prevOrientations):
        """ Teleports the blocks into a specified position / orientation """
        for i in range(len(positions)):
            # If the position/orientation did not change, skip moving this block
            if prevPositions is not None and prevOrientations is not None and len(positions)==len(prevPositions) and np.abs((np.array(positions[i])-np.array(prevPositions[i]))).sum() < 1e-6:
                continue
            
            state = [(positions[i][j] + self.shift[k]) * (self.grid_len/2) for k, j in enumerate(self.ordering)]
            state[1] = -state[1]
            orient = orientations[i]
            
            # Convert quaternion to euler angles
            if len(orient)==4:
                orient = self.quat2euler(orient)
            
            self.robot.reposition_object_at_list_index_to_location(state, orient, i)
            
        # Move unused objects out of the camera frame
        for i in range (self.num_obj-len(positions)):
            out_state, out_ori = [-0.75 - i*(self.side_len+0.01), 0, self.side_len], [0, 0, 0]
            self.robot.reposition_object_at_list_index_to_location(out_state, out_ori, len(positions)+i)
        return 0
        
        
    def captureAndSaveImages(self, filename, colorDir, depthDir, colorHMDir, depthHMDir, debug=False):
        """ Returns an nparray with an RGB image """
        _ , color_heightmap, depth_heightmap, _ , color_img, depth_img = self.robot.get_camera_data(return_heightmaps=True, color_median_filter_size=0)
        
        # Save the regular camera captures (iso side view)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        resized_img = cv2.resize(color_img.copy(), (480, 360), interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(colorDir, filename), resized_img)
        # cv2.imwrite(os.path.join(colorDir, filename), color_img)
        depth_img = np.round(depth_img * 10000).astype(np.uint16) # Save depth in 1e-4 meters
        resized_depth_img = cv2.resize(depth_img.copy(), (480, 360), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(depthDir, filename), depth_img)
        
        # Save the heightmaps (top view)
        color_heightmap = cv2.cvtColor(color_heightmap, cv2.COLOR_RGB2BGR)
        color_heightmap = cv2.flip(color_heightmap, 0)
        if debug:
            original_depth_heightmap = depth_heightmap.copy()
        cv2.imwrite(os.path.join(colorHMDir, filename), color_heightmap)
        depth_heightmap = np.round(depth_heightmap * 100000).astype(np.uint16) # Save depth in 1e-5 meters
        depth_heightmap = cv2.flip(depth_heightmap, 0)
        depth_heightmap_path = os.path.join(depthHMDir, filename)
        cv2.imwrite(depth_heightmap_path, depth_heightmap)
        if debug:
            converted_depth_heightmap = depth_heightmap.astype(np.float32) / 100000
            saved_reloaded_depth_heightmap = np.array(cv2.imread(depth_heightmap_path, cv2.IMREAD_ANYDEPTH)).astype(np.float32) / 100000
            import matplotlib.pyplot as plt
            f = plt.figure()
            f.add_subplot(1,3, 1)
            plt.imshow(original_depth_heightmap)
            f.add_subplot(1,3, 2)
            # f.add_subplot(1, 2, 1)
            plt.imshow(converted_depth_heightmap)
            f.add_subplot(1,3, 3)
            plt.imshow(saved_reloaded_depth_heightmap)
            plt.show(block=True)
        
        return 0



def main(args):
    # Create the image output folders if they don't already exist
    folderList = [args.colorimg_folder, args.depthimg_folder, args.colorHm_folder, args.depthHm_folder]
    for folder in folderList:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Instantiate the dataset reader object
    dataset_reader = DatasetReader(args.train_path,
                                   args.val_path,
                                   None,
                                   batch_by_line=True,
                                   batch_size = 1)
    print(f"Reading data from {args.train_path}, this may take a few minutes.")
    
    # Load from the train set
    vocab = dataset_reader.read_data("train")
    train_data = dataset_reader.data["train"]
    print("Done reading trajectory data")
    
    imc = None
    pos = None
    rot = None
    
    for b, trajectory in tqdm(enumerate(train_data)):
    
        # Store previous position
        ppos = pos
        prot = rot
        
        # Assign block positions and orientations from data instance
        pos = trajectory.previous_positions
        rot = trajectory.previous_rotations
        
        # The dataset reader gives each note as an instance, but images should be generated per state.
        # A state can have several sequential notes, thus we can skip if diff == 0
        if ppos is not None:
            if len(pos[0])==len(ppos[0]) and np.abs((np.array(pos)-np.array(ppos))).sum() < 1e-6:
                continue
        
        # Load the setup
        if imc is None:
            imc = BlockSetter(pos, rot, args.blocks_path, args.num_blocks, args.offset)
        else:
            imc.load_setup(pos[0], rot[0], ppos[0], prot[0])
        
        # Capture the image and save it
        imc.captureAndSaveImages(trajectory.images[0], args.colorimg_folder, args.depthimg_folder, args.colorHm_folder, args.depthHm_folder)
        
        # Repeat the process with the next state
        npos = trajectory.next_positions
        nrot = trajectory.next_rotations
        imc.load_setup(npos[0], nrot[0], pos[0], rot[0])
        imc.captureAndSaveImages(trajectory.images[1], args.colorimg_folder, args.depthimg_folder, args.colorHm_folder, args.depthHm_folder)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default = "blocks_data/trainset_v2.json", help="path to train data")
    parser.add_argument("--val-path", default = "blocks_data/devset.json", type=str, help = "path to dev data")
    parser.add_argument("--num-blocks", type=int, default=20)
    parser.add_argument("--simulator", default="CoppeliaSim", help="simulator software name")
    # parser.add_argument("--trajectory-path", type=str, default = "blocks_data/singleset.json", help="trajectory motion of blocks")
    parser.add_argument("--blocks-path", type=str, default = "objects/bisk_blocks", help="trajectory values")
    parser.add_argument("--colorimg-folder", type=str, default = "data/color-images", help="output folder for color images")
    parser.add_argument("--depthimg-folder", type=str, default = "data/depth-images", help="output folder for depth images")
    parser.add_argument("--colorHm-folder", type=str, default = "data/color-heightmaps", help="output folder for color heightmaps")
    parser.add_argument("--depthHm-folder", type=str, default = "data/depth-heightmaps", help="output folder for depth heightmaps")
    parser.add_argument("--quaternions", action='store_true')
    parser.add_argument("--offset", type=float, nargs=3, default=[0.15, 0.0, 0.0],  help="enter three float values to shift blocks on plane")
    args = parser.parse_args()
    main(args)
