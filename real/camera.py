#!/usr/bin/env python

# Ros libraries
import roslib
import rospy

import socket
import numpy as np
import cv2
import os
import time
import struct
from .ros_camera import ROSCamera
# Not using the primesense_sensor package and switch to ROS with rectified images
# try:
#     import primesense_sensor
#     from primesense_sensor as PrimesenseSenor
# except ImportError:
#     print('primesense_sensor is not available!'
#           'The primesense_sensor package can be installed from: '
#           'https://berkeleyautomation.github.io/perception/install/install.html')
#     primesense_sensor = None

class Camera(object):

    def __init__(self, model_name='primesense', camera_intrinsic_folder='~/src/real_good_robot/real/camera_param', calibrate=False):
        
        self.model_name = model_name

        if self.model_name is not 'primesense':
            # Data options (change me)
            self.im_height = 480
            self.im_width = 640
            self.tcp_host_ip = '127.0.0.1'
            self.tcp_port = 50000
            self.buffer_size = 4098 # 4 KiB

            # Connect to server
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            self.intrinsics = None
            self.get_data()

        else:
            # Lauch ROS to get the image
            '''Initializes and cleanup ros node'''
            print('Before you run test_ros_images.py, make sure you have first started the ROS node for reading the primesense sensor data:'
            ' roslaunch openni2_launch openni2.launch depth_registration:=true')
            self.camera = ROSCamera(calibrate=calibrate)
            rospy.init_node('ros_camera', anonymous=True)
            time.sleep(1)  
            # Camera matrix K
            camera_matrix_txt = os.path.join(os.path.expanduser(camera_intrinsic_folder), 'CameraMatrixRGB.dat')
            self.intrinsics = np.loadtxt(camera_matrix_txt)


    def get_data(self, undistort=False):
        
        if self.model_name is not 'primesense':
            # Ping the server with anything
            self.tcp_socket.send(b'asdf')

            # Fetch TCP data:
            #     color camera intrinsics, 9 floats, number of bytes: 9 x 4
            #     depth scale for converting depth from uint16 to float, 1 float, number of bytes: 4
            #     depth image, self.im_width x self.im_height uint16, number of bytes: self.im_width x self.im_height x 2
            #     color image, self.im_width x self.im_height x 3 uint8, number of bytes: self.im_width x self.im_height x 3
            data = b''
            while len(data) < (10*4 + self.im_height*self.im_width*5):
                data += self.tcp_socket.recv(self.buffer_size)

            # Reorganize TCP data into color and depth frame
            self.intrinsics = np.fromstring(data[0:(9*4)], np.float32).reshape(3, 3)
            depth_scale = np.fromstring(data[(9*4):(10*4)], np.float32)[0]
            depth_img = np.fromstring(data[(10*4):((10*4)+self.im_width*self.im_height*2)], np.uint16).reshape(self.im_height, self.im_width)
            color_img = np.fromstring(data[((10*4)+self.im_width*self.im_height*2):], np.uint8).reshape(self.im_height, self.im_width, 3)
            depth_img = depth_img.astype(float) * depth_scale

        else:
            # Get frame
            color_img, depth_img, _ = self.camera.frames()
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        return color_img, depth_img
    
    def subscribe_aruco_tf(self):
        self.camera.subscribe_aruco_tf()
    
    def get_aruco_tf(self):
        aruco_tf, aruco_img = self.camera.aruco()
        return aruco_tf, aruco_img
