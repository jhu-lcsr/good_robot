#! /usr/bin/env python3

'''

Test for the PrimeSense Sensor for the Real Good Robot project
Author: Hongtao Wu
Sep 29, 2019

The primesense_sensor package can be installed from here:
https://berkeleyautomation.github.io/perception/install/install.html
'''

from primesense_sensor import PrimesenseSensor
import cv2
import numpy as np

def main():
    PS = PrimesenseSensor()
    PS.start()

    # while True:
    color_img, depth_img, _ = PS.frames()
    # print(f'color_img type: {color_img.raw_data.dtype}')
    # print(f'depth_img type: {depth_img.raw_data.dtype}')

    color_img_raw = cv2.cvtColor(color_img.raw_data, cv2.COLOR_BGR2RGB)

    cv2.imshow('test_color.png', color_img_raw)
    cv2.imshow('test_depth.png', depth_img.raw_data)
    cv2.waitKey(0)

    import ipdb; ipdb.set_trace()

    print(depth_img.raw_data)

    # Camera distortion test
    RGB_K = np.loadtxt('/home/costar/src/real_good_robot/real/camera_param/CameraMatrixRGB.dat')
    RGB_D = np.loadtxt('/home/costar/src/real_good_robot/real/camera_param/DistortionCoefficientRGB.dat')

    Depth_K = np.loadtxt('/home/costar/src/real_good_robot/real/camera_param/CameraMatrixDepth.dat')
    Depth_D = np.loadtxt('/home/costar/src/real_good_robot/real/camera_param/CameraMatrixDepth.dat')

    color_img_undistort = cv2.undistort(color_img_raw, RGB_K, RGB_D)
    depth_img_undistort = cv2.undistort(depth_img.raw_data, Depth_K, Depth_D)
    
    cv2.imshow('color_undistort.png', color_img_undistort)
    cv.imshow('depth_undistort.png', depth_img_undistort)
    cv2.imshow()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
