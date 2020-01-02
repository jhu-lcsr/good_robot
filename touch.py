#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from robot import Robot
from real.camera import Camera
import threading

class HumanControlOfRobot(object):
    """Creates a color and depth opencv window from the robot camera, gets human keyboard/click, and moves the robot.

    Keyboard Controls:
        'c': set self.stop to True indicating it is time to exit the program.
        'a': autonomous mode, sets self.human_control = False, clicks will have no effect (unless move_robot=False).
        'h': human control mode, sets self.human_control = True, clicks will move the robot (unless move_robot=False).
        'g': set self.action = 'grasp', left click in the 'color' image window will do a grasp action.
        'p': set self.action = 'place', left click will do a place action.
        's': set self.action = 'push', left click will slide the gripper across the ground, aka a push action.
        't': set self.action = 'touch', left click will do a touch action (go to a spot and stay there).
        '1-9': Set the gripper rotation orientation at 45 degree increments, starting at the angle 0. Default is '5'.
        'b': set self.action = box, left click will move the robot to go get the box and dump the objects inside.
        '[': set self.robot.place_task = False, a successful grasp will immediately drop objects in the box.
        ']': set self.robot.place_task = True, a successful grasp will hold on to objects so the robot can place them.
        ' ': print the current robot cartesian position with xyz and axis angle and the current joint angles.

    Member Variables:

        self.stop: if True shut down your program, pressing 'c' on the keyboard sets this variable to True.
    """
    def __init__(self, robot=None, action='touch', human_control=True, mutex=None, move_robot=True):
        self.stop = False
        self.print_state_count = 0
        self.tool_orientation = [0.0, np.pi, 0.0] # Real Good Robot
        self.human_control = human_control
        self.move_robot = move_robot
        self.action = action
        self.click_count = 0
        self.target_position = None
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
            if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
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
                with self.mutex:
                    self.target_position = target_position
                    self.click_count += 1
                    if self.action == 'touch':
                        # Move the gripper up a bit to protect the gripper (Real Good Robot)
                        self.target_position[-1] += 0.17
                        def move_to():
                            # global self.mutex
                            with self.mutex:
                                robot.move_to(target_position, self.tool_orientation)
                        if self.move_robot:
                            t = threading.Thread(target=move_to)
                            t.start()
                    elif self.action == 'grasp':
                        if not robot.place_task or (robot.place_task and not self.grasp_success):
                            def grasp():
                                # global self.grasp_success, self.grasp_color_success, self.mutex
                                with self.mutex:
                                    self.grasp_success, self.grasp_color_success = robot.grasp(target_position, self.grasp_angle * np.pi / 4)
                            if self.move_robot:
                                t = threading.Thread(target=grasp)
                                t.start()
                        else:
                            def place():
                                # global self.grasp_success, self.mutex
                                with self.mutex:
                                    robot.place(target_position, self.grasp_angle * np.pi / 4)
                                    self.grasp_success = False
                            if self.move_robot:
                                t = threading.Thread(target=place)
                                t.start()

                    elif self.action == 'box':
                        t = threading.Thread(target=lambda: robot.restart_real())
                        t.start()
                    elif self.action == 'push':
                        self.target_position[-1] += 0.01
                        t = threading.Thread(target=lambda: robot.push(target_position, self.grasp_angle * np.pi / 4))
                        t.start()
                    elif self.action == 'place':
                        self.target_position[-1] += 0.01
                        t = threading.Thread(target=lambda: robot.place(target_position, self.grasp_angle * np.pi / 4))
                        t.start()
                    
                    self.target_position = target_position

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
        cv2.imshow('depth', self.camera_depth_img)
        
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
            with self.mutex:
                # Mode for stacking blocks
                self.robot.place_task = True
                self.print_task()
        elif key == ord('['):
            with self.mutex:
                # Mode for grasping to hold and then place
                self.robot.place_task = False
                self.print_task()
        elif key == ord(' '):
            with self.mutex:
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
    
    def run(self):
        """ Blocking call that repeatedly calls run_one()
        """
        while not hcr.stop:
            hcr.run_one()
        
    def get_action(self, camera_color_img=None, camera_depth_img=None, prev_click_count=None, block=True):
        """ Get a human specified action
        # Arguments
            camera_color_img: show the human user a specific color image
            camera_depth_img: show the human user a specific depth image
            prev_click_count: pass the click count you saw most recently, used to determine if the user clicked in between calls to get_action.
            block: when True this function will loop and get keypresses via run_one() until a click is received, when false it will just immediately return the current state.
        # Returns
            [action_name, target_position, grasp_angle, cur_click_count, camera_color_img, camera_depth_img]
        """
        running = True
        if prev_click_count is None:
            with self.mutex:
                prev_click_count = self.click_count
        while running:
            self.run_one(camera_color_img, camera_depth_img)
            with self.mutex:
                cur_click_count = self.click_count
                action = self.action
                target_position = self.target_position
                grasp_angle = self.grasp_angle
                if running:
                    running = not self.stop

            if not block:
                running = False
            elif cur_click_count > prev_click_count:
                running = False
        if camera_color_img is None:
            with self.mutex:
                camera_color_img = self.camera_color_img
                camera_depth_img = self.camera_depth_img
        return action, target_position, grasp_angle, cur_click_count, camera_color_img, camera_depth_img
    
    def __del__(self):
        cv2.destroyAllWindows()

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
                False, None, None, place=True)
    hcr = HumanControlOfRobot(robot, action=action)
    hcr.run()
    # while not hcr.stop:
    #     # hcr.run_one()
    #     hcr.get_action()
    # cv2.destroyAllWindows()
