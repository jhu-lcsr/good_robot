# Run by itself, e.g. python tmp_gripper.py
# Moves to home position (twice, once on class init, and once with move_joints)
# Then moves to absolute commanded ,w


import time
import numpy as np
from robot import Robot
import socket
import struct

# User options (change me)
# --------------- Setup options ---------------
# tcp_host_ip = '100.127.7.223' # IP and port to robot arm as TCP client (UR5)
tcp_host_ip = "10.75.15.91"
tcp_port = 30002
# rtc_host_ip = '100.127.7.223' # IP and port to robot arm as real-time client (UR5)
rtc_host_ip = "10.75.15.91"
rtc_port = 30003

# Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
workspace_limits = np.asarray([[0.3, 0.748], [0.05, 0.4],
                               [-0.2 + 0.4, -0.1 + 0.4]])
# ---------------------------------------------

print('Connecting to robot...')
# second arg: is_sim (None or False)
robot = Robot(False, False, None, workspace_limits,
              tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
              False, None, None)
# robot.close_gripper()
print('Gripper closed')
time.sleep(0.3)
# robot.open_gripper()
print('Gripper opened')
time.sleep(0.3)

robot.joint_acc = 0.1
robot.joint_vel = 0.1

# tool_orientation = [-np.pi/2, 0, 0]  # [0,-2.22,2.22] # [2.22,2.22,0]
# calib_home = [-np.pi, -np.pi/2, np.pi/2, 0, np.pi/2, np.pi]
# python 0.47 = 0.07 on the pendant
# workspace_fixoffset = [0.5, 0.2, 0.47, -1.22, 1.19, -1.17]
home_in_deg = np.array(
    [-197, -105, 130, -110, -90, -30]) * 1.0
home_joint_config = np.deg2rad(home_in_deg)

# --------------------- NOTE: Change me!! -------------------
commanded = [0.348, 0.0000, 0.268, 2.107, -2.200, -0.065]

print('!-------- Moving joints to', home_joint_config, ' ----------\n\n')
robot.move_joints(home_joint_config)
time.sleep(1)

print('!--------------------- Moved to calib home. next: ------------')
print('!----- Moving l (pos) to', commanded[0:3], commanded[3:], ' ---\n\n')
robot.move_to(commanded[0:3], commanded[3:])
time.sleep(1)
