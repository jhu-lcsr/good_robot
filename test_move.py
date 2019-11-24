from robot import Robot
import numpy as np
import time

r = Robot(is_sim=False, tcp_host_ip='192.168.1.155', tcp_port=30002)

# tool_orientation = [0.0, 0.0, 0.0] # Real Good Robot
# above_bin_waypoint = [0.3, 0.0,  0.8]
# r.move_to(above_bin_waypoint, tool_orientation)
# time.sleep(.1)

tool_pos = [0.6, -0.1, 0.4]
tool_orientation = [0.0, np.pi, 0.0]
r.move_to(tool_pos, tool_orientation)

grasp_orientation = [1, 0]
tool_rotation_angle = np.pi/2 / 2
tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
print(tool_orientation)
r.move_to(tool_pos, tool_orientation)

# tool_orientation = [np.pi/2, np.pi/2, 0.0]
# r.move_to(tool_pos, tool_orientation)