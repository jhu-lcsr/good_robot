#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from robot import Robot
from real.camera import Camera
import threading
    

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
tool_orientation = [0.0, np.pi, 0.0] # Real Good Robot

# workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
# tool_orientation = [2.22,-2.22,0]
# ---------------------------------------------


# Move robot to home pose
robot = Robot(False, None, None, workspace_limits,
              tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
              False, None, None)
robot.open_gripper()

# Slow down robot
# robot.joint_acc = 1.4
# robot.joint_vel = 1.05
grasp_angle = 4.0
grasp_success, grasp_color_success = False, False
mutex = threading.Lock()
# Callback function for clicking on OpenCV window
click_point_pix = ()
# camera_color_img, camera_depth_img = robot.get_camera_data()
def mouseclick_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global camera, robot, click_point_pix, action, grasp_angle, grasp_success, grasp_color_success, mutex
        click_point_pix = (x,y)

        # Get click point in camera coordinates
        click_z = camera_depth_img[y][x] * robot.cam_depth_scale * 1000 # unit from m -> mm
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
        print(target_position, tool_orientation)
        
        with mutex:
            if action == 'touch':
                # Move the gripper up a bit to protect the gripper (Real Good Robot)
                target_position[-1] += 0.17
                def move_to():
                    global mutex
                    with mutex:
                        robot.move_to(target_position, tool_orientation)
                t = threading.Thread(target=move_to)
                t.start()
            elif action == 'grasp':
                if not robot.place_task or (robot.place_task and not grasp_success):
                    def grasp():
                        global grasp_success, grasp_color_success, mutex
                        with mutex:
                            grasp_success, grasp_color_success = robot.grasp(target_position, grasp_angle * np.pi / 4)
                    t = threading.Thread(target=grasp)
                    t.start()
                else:
                    def place():
                        global grasp_success, mutex
                        with mutex:
                            robot.place(target_position, grasp_angle * np.pi / 4)
                            grasp_success = False
                    t = threading.Thread(target=place)
                    t.start()

            elif action == 'box':
                t = threading.Thread(target=lambda: robot.restart_real())
                t.start()
            elif action == 'push':
                target_position[-1] += 0.01
                t = threading.Thread(target=lambda: robot.push(target_position, grasp_angle * np.pi / 4))
                t.start()
            elif action == 'place':
                target_position[-1] += 0.01
                t = threading.Thread(target=lambda: robot.place(target_position, grasp_angle * np.pi / 4))
                t.start()


# Show color and depth frames
cv2.namedWindow('depth')
cv2.namedWindow('color')
cv2.setMouseCallback('color', mouseclick_callback)
def print_task():
    global robot
    print('place task') if robot.place_task else print('push grasp task')

print_task()
i = 0
while True:
    camera_color_img, camera_depth_img = robot.get_camera_data()
    if len(click_point_pix) != 0:
        camera_color_img = cv2.circle(camera_color_img, click_point_pix, 7, (0,0,255), 2)
    camera_color_img = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
    cv2.imshow('color', camera_color_img)
    cv2.imshow('depth', camera_depth_img)
    
    key = cv2.waitKey(1)
    # Configure the system
    # Numbers 1-9 are orientations of the gripper
    # t is touch mode, where the robot will go to the clicked spot

    if key == ord('1'):
        grasp_angle = 0.0
    elif key == ord('2'):
        grasp_angle = 1.0
    elif key == ord('3'):
        grasp_angle = 2.0
    elif key == ord('4'):
        grasp_angle = 3.0
    elif key == ord('5'):
        grasp_angle = 4.0
    elif key == ord('6'):
        grasp_angle = 5.0
    elif key == ord('7'):
        grasp_angle = 6.0
    elif key == ord('8'):
        grasp_angle = 7.0
    elif key == ord('9'):
        grasp_angle = 8.0
    elif key == ord('t'):
        action = 'touch'
    elif key == ord('g'):
        action = 'grasp'
    elif key == ord('s'):
        action = 'push'
    elif key == ord('p'):
        action = 'place'
    elif key == ord('b'):
        action = 'box'
    elif key == ord(']'):
        # Mode for stacking blocks
        robot.place_task = True
        print_task()
    elif key == ord('['):
        # Mode for grasping to hold and then place
        robot.place_task = False
        print_task()
    elif key == ord(' '):
        # print the robot state
        i += 1
        state_data = robot.get_state()
        actual_tool_pose = robot.parse_tcp_state_data(state_data, 'cartesian_info')
        joint_position = robot.parse_tcp_state_data(state_data, 'joint_data')
        robot_state = 'cart_pose: ' + str(actual_tool_pose) + ' joint pos: ' + str(joint_position)
        print(str(i) + ' ' + robot_state)
    elif key == ord('c'):
        break

cv2.destroyAllWindows()
