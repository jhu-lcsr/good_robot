#!/usr/bin/env python
"""OpenCV feature detectors with ros CompressedImage Topics in python.

This example subscribes to a ros topic containing sensor_msgs 
CompressedImage. It converts the CompressedImage into a numpy.ndarray, 
then detects and marks features in that image. It finally displays 
and publishes the new image - again as CompressedImage topic.

Based on code at http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber by Simon Haller <simon.haller at uibk.ac.at>
"""
__author__ =  'Andrew Hundt <ATHundt@gmail.com>'
__version__=  '0.1'
__license__ = 'BSD'
# Python libs
import sys, time

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from sensor_msgs.msg import CameraInfo
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from threading import Lock
import message_filters
from real.ros_camera import ROSCamera

VERBOSE=False

def main(args):
    '''Initializes and cleanup ros node'''
    print('Before you run test_ros_images.py, make sure you have first started the ROS node for reading the primesense sensor data:'
          ' roslaunch openni2_launch openni2.launch depth_registration:=true')
    camera = ROSCamera()
    rospy.init_node('ros_camera', anonymous=True)
    try:
        time.sleep(1)
        # i = 0
        while True:
            # print(i)
            # rospy.spin_one()
            rgb, depth, _ = camera.frames()
            if rgb is not None:
                cv2.imshow('color.png', rgb)
            if depth is not None:
                # Multiply just so you can see the values, don't do that with real data
                depth *= 100
                cv2.imshow('depth.png', depth)
            cv2.waitKey(1)
            # i += 1

    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)