#!/usr/bin/env python

import numpy as np
import time
import os
import sys
from robot import Robot


# User options (change me)
# --------------- Setup options ---------------
tcp_host_ip = "10.75.15.91"  # IP and port to robot arm as TCP client (UR5)
tcp_port = 30002
# Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
# NOTE: original
# workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]])

# NOTE: mine
# workspace_limits = np.asarray(
# [[0.4, 0.75], [-0.25, 0.15], [-0.15 + 0.4, -0.1 + 0.4]])
# workspace_limits = np.asarray(
# [[0.300, 0.500], [-0.250, 0.150], [0.200, 0.300]])
workspace_limits = np.asarray(
    # [[0.300, 0.600], [-0.250, 0.150], [0.200, 0.400]])
    [[0.300, 0.600], [-0.250, 0.150], [0.200, 0.300]])
# ---------------------------------------------

# Initialize robot and move to home pose
robot = Robot(False, False, None, workspace_limits,
              tcp_host_ip, tcp_port, None, None,
              False, None, None)

# Repeatedly grasp at middle of workspace
grasp_position = np.sum(workspace_limits, axis=1) / 2
grasp_position[0] -= 0.100
grasp_position[2] = 0.25  # NOTE this sets z position!

# NOPE grasp_position[2] = -0.25 # extra NOPE this would be through the table for me

# grasp_position[0] = 86 * 0.002 + workspace_limits[0][0]
# grasp_position[1] = 120 * 0.002 + workspace_limits[1][0]
# grasp_position[2] = workspace_limits[2][0]

# try:
while True:
    # print('\n !------Attempting grasp at pos:  ', grasp_position, ' ---')
    # robot.grasp(grasp_position, 11 * np.pi / 8, workspace_limits)
    # time.sleep(0.1)
    # print('!----Grasp completed')
    # robot.close_gripper()
    # time.sleep(0.1)

    # robot.open_gripper()
    try:
        prompt1 = raw_input("Place object in grasp, then 'y' to close grasp: ")
        if str(prompt1) == 'y':
            time.sleep(0.01)
            robot.close_gripper()
            # time.sleep(0.01)
        prompt2 = raw_input("Stand clear, then 'yes' to throw: ")
        if str(prompt2) == 'yes':
            time.sleep(0.05)

            print('!----Throw started')
            robot.r.throw()
            print('!----Throw completed')

        # # robot.push(push_position, 0, workspace_limits)
        # # robot.restart_real()

        # Repeatedly move to workspace corners
        # while True:
        '''
        print('Attempting to debug.')
        # move in a sqaure
        robot.move_to([workspace_limits[0][0], workspace_limits[1][0],
                       workspace_limits[2][0]], None, acc_scaling=1,
                      vel_scaling=1)
        time.sleep(0.2)

        robot.move_to([workspace_limits[0][0], workspace_limits[1][1],
                       workspace_limits[2][0]], None)
        time.sleep(0.2)
        robot.close_gripper()

        robot.move_to([workspace_limits[0][1], workspace_limits[1][1],
                       workspace_limits[2][0]], None)
        time.sleep(0.2)

        robot.move_to([workspace_limits[0][1], workspace_limits[1][0],
                       workspace_limits[2][0]], None)
        time.sleep(0.2)
        robot.open_gripper()
        '''

    except KeyboardInterrupt:
        print('closing robot')
        sys.exit()
    # except Exception as e:  # RobotError, ex:
        # print("Robot could not execute move (emergency stop for example), do something", e)
        # robot.r.close()
        # # print('exiting')
        # # sys.exit()
        # print('exiting again')
        # os._exit(1)
