import time
import numpy as np
from robot import Robot

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
time.sleep(1)
robot.open_gripper()
print('Gripper opened')
time.sleep(1)

robot.close_gripper()
print('Gripper closed')
time.sleep(1)
robot.open_gripper()
print('Gripper opened')
time.sleep(1)
