import time
import numpy as np
from robot import Robot
import socket
import struct

# User options (change me)
# --------------- Setup options ---------------
# tcp_host_ip = '100.127.7.223' # IP and port to robot arm as TCP client (UR5)
tcp_host_ip = "10.75.15.94"
tcp_port = 30002
# rtc_host_ip = '100.127.7.223' # IP and port to robot arm as real-time client (UR5)
rtc_host_ip = "10.75.15.94"
rtc_port = 30003

# Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
workspace_limits = np.asarray([[0.3, 0.748], [0.05, 0.4], [-0.2, -0.1]])
# ---------------------------------------------

print('Connecting to robot...')
# second arg: is_sim (None or False)
robot = Robot(False, False, None, workspace_limits,
              tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
              False, None, None)
robot.close_gripper()
print('Gripper closed')
time.sleep(0.3)
robot.open_gripper()
print('Gripper opened')
time.sleep(0.3)


robot.joint_acc = 0.1
robot.joint_vel = 0.1

# tool_orientation = [-np.pi/2, 0, 0]  # [0,-2.22,2.22] # [2.22,2.22,0]

calib_home = [-np.pi, -np.pi/2, np.pi/2, 0, np.pi/2, np.pi]
workspace_center = [0.5, 0.2, -0.1, 0, np.pi/2, np.pi]
workspace_relative = [0.1, 0.1, 0.1, 0.01, 0.01, 0.01]
workspace_axis_angle = [0.5, 0.2, -0.1,
                        np.pi*(3/4.), -np.pi*(3/4), np.pi*(3/4)]

# --------------------- NOTE: Change me!! -------------------
commanded = workspace_relative

tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_socket.connect((tcp_host_ip, tcp_port))
tcp_command = "get_target_tcp_pose()"
tcp_socket.send(str.encode(tcp_command))
state_data = tcp_socket.recv(4096)
tcp_socket.close()

data_bytes = bytearray()
data_bytes.extend(state_data)
data_length = struct.unpack("!i", data_bytes[0:4])[0]
robot_message_type = data_bytes[4]
print('msg type', robot_message_type)
print('length', data_length)
print('data', data_bytes)


print('!-------- Moving joints to', calib_home, ' ----------\n\n')
robot.move_joints(calib_home)
time.sleep(1)

print('!--------------------- Moved to calib home. next: ------------')
print('!----- Moving l (pos) to', commanded[0:3], commanded[3:], ' ---\n\n')
# robot.move_to(commanded[0:3], commanded[3:])
time.sleep(1)
