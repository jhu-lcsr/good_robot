#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from robot import Robot
from real.camera import Camera
import threading

class HumanControlOfRobot(object):

    def __init__(self, robot=None, action='touch', human_control=True, mutex=None):
        self.stop = False
        self.print_state_count = 0
        self.tool_orientation = [0.0, np.pi, 0.0] # Real Good Robot
        self.human_control = human_control
        if robot is None:

            # workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
            # self.tool_orientation = [2.22,-2.22,0]
            # ---------------------------------------------
            # Move robot to home pose
            self.robot = Robot(False, None, None, workspace_limits,
                               tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                               False, None, None)
            robot.open_gripper()
        else:
            self.robot = robot

        # Slow down robot
        # robot.joint_acc = 1.4
        # robot.joint_vel = 1.05
        self.grasp_angle = 4.0
        self.grasp_success, self.grasp_color_success = False, False
        if mutex is None:
            self.mutex = threading.Lock()
        # Callback function for clicking on OpenCV window
        self.click_point_pix = ()
        
        self.camera_color_img, self.camera_depth_img = robot.get_camera_data()
        def mouseclick_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # global camera, robot, self.click_point_pix, action, self.grasp_angle, self.grasp_success, self.grasp_color_success, self.mutex
                self.click_point_pix = (x,y)

                # Get click point in camera coordinates
                click_z = self.camera_depth_img[y][x] * robot.cam_depth_scale * 1000 # unit from m -> mm
                click_x = np.multiply(x-robot.cam_intrinsics[0][2],click_z/robot.cam_intrinsics[0][0])
                click_y = np.multiply(y-robot.cam_intrinsics[1][2],click_z/robot.cam_intrinsics[1][1])
                if click_z == 0:
                    return
                click_point = np.asarray([click_x,click_y,click_z]) / 1000  # Convert from unit from mm to m
                click_point.shape = (3,1)

                # Convert camera to robot coordinates
                # camera2robot = np.linalg.inv(robot.cam_pose)
                camera2robot = robot.cam_pose  # The transformation matrix is from meter to meter
                target_position = np.dot(camera2robot[0:3,0:3],click_point) + camera2robot[0:3,3:]

                target_position = target_position[0:3,0]
                print(target_position, self.tool_orientation)
                
                if not self.human_control:
                    print('Human Control is disabled, press h for human control mode, a for autonomous mode')
                    return
                with self.mutex:
                    if action == 'touch':
                        # Move the gripper up a bit to protect the gripper (Real Good Robot)
                        target_position[-1] += 0.17
                        def move_to():
                            # global self.mutex
                            with self.mutex:
                                robot.move_to(target_position, self.tool_orientation)
                        t = threading.Thread(target=move_to)
                        t.start()
                    elif action == 'grasp':
                        if not robot.place_task or (robot.place_task and not self.grasp_success):
                            def grasp():
                                # global self.grasp_success, self.grasp_color_success, self.mutex
                                with self.mutex:
                                    self.grasp_success, self.grasp_color_success = robot.grasp(target_position, self.grasp_angle * np.pi / 4)
                            t = threading.Thread(target=grasp)
                            t.start()
                        else:
                            def place():
                                # global self.grasp_success, self.mutex
                                with self.mutex:
                                    robot.place(target_position, self.grasp_angle * np.pi / 4)
                                    self.grasp_success = False
                            t = threading.Thread(target=place)
                            t.start()

                    elif action == 'box':
                        t = threading.Thread(target=lambda: robot.restart_real())
                        t.start()
                    elif action == 'push':
                        target_position[-1] += 0.01
                        t = threading.Thread(target=lambda: robot.push(target_position, self.grasp_angle * np.pi / 4))
                        t.start()
                    elif action == 'place':
                        target_position[-1] += 0.01
                        t = threading.Thread(target=lambda: robot.place(target_position, self.grasp_angle * np.pi / 4))
                        t.start()

        # Show color and depth frames
        cv2.namedWindow('depth')
        cv2.namedWindow('color')
        cv2.setMouseCallback('color', mouseclick_callback)

        self.print_task()
    
    def print_task(self):
        # global robot
        print('place task') if robot.place_task else print('push grasp task')

    def run_one(self, camera_color_img=None, camera_depth_img=None):
        if camera_color_img is None:
            self.camera_color_img, self.camera_depth_img = robot.get_camera_data()
        else:
            self.camera_color_img = camera_color_img
            self.camera_depth_img = camera_depth_img
        if len(self.click_point_pix) != 0:
            self.camera_color_img = cv2.circle(self.camera_color_img, self.click_point_pix, 7, (0,0,255), 2)
        self.camera_color_img = cv2.cvtColor(self.camera_color_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('color', self.camera_color_img)
        cv2.imshow('depth', self.camera_color_img)
        
        key = cv2.waitKey(1)
        # Configure the system
        # Numbers 1-9 are orientations of the gripper
        # t is touch mode, where the robot will go to the clicked spot

        if key == ord('1'):
            self.grasp_angle = 0.0
        elif key == ord('2'):
            self.grasp_angle = 1.0
        elif key == ord('3'):
            self.grasp_angle = 2.0
        elif key == ord('4'):
            self.grasp_angle = 3.0
        elif key == ord('5'):
            self.grasp_angle = 4.0
        elif key == ord('6'):
            self.grasp_angle = 5.0
        elif key == ord('7'):
            self.grasp_angle = 6.0
        elif key == ord('8'):
            self.grasp_angle = 7.0
        elif key == ord('9'):
            self.grasp_angle = 8.0
        elif key == ord('t'):
            self.action = 'touch'
        elif key == ord('g'):
            self.action = 'grasp'
        elif key == ord('s'):
            self.action = 'push'
        elif key == ord('p'):
            self.action = 'place'
        elif key == ord('b'):
            self.action = 'box'
        elif key == ord(']'):
            # Mode for stacking blocks
            self.robot.place_task = True
            self.print_task()
        elif key == ord('['):
            # Mode for grasping to hold and then place
            self.robot.place_task = False
            self.print_task()
        elif key == ord(' '):
            # print the robot state
            self.print_state_count += 1
            state_data = robot.get_state()
            actual_tool_pose = robot.parse_tcp_state_data(state_data, 'cartesian_info')
            joint_position = robot.parse_tcp_state_data(state_data, 'joint_data')
            robot_state = 'cart_pose: ' + str(actual_tool_pose) + ' joint pos: ' + str(joint_position)
            print(str(self.print_state_count) + ' ' + robot_state)
        elif key == ord('c'):
            self.stop = True
        elif key == ord('h'):
            self.human_control = True
        elif key == ord('a'):
            self.human_control = False

if __name__ == '__main__':

    # User options (change me)
    # --------------- Setup options ---------------
    tcp_host_ip = '192.168.1.155' # IP and port to robot arm as TCP client (UR5)
    tcp_port = 30002
    rtc_host_ip = '192.168.1.155' # IP and port to robot arm as real-time client (UR5)
    rtc_port = 30003
    # action = 'touch'
    action = 'grasp'

    if action == 'touch':
        # workspace_limits = np.asarray([[0.5, 0.75], [-0.3, 0.1], [0.17, 0.3]]) # Real Good Robot
        workspace_limits = None
    elif action == 'grasp':
        workspace_limits = None
    else:
        raise NotImplementedError

    # Move robot to home pose
    robot = Robot(False, None, None, workspace_limits,
                tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                False, None, None)
    hcr = HumanControlOfRobot(robot, action=action)
    while not hcr.stop:
        hcr.run_one()
    cv2.destroyAllWindows()
