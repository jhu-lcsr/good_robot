import socket
import select
import struct
import time
import os
import numpy as np
import utils
import serial
import binascii

from pyUR import URcomm

from simulation import vrep
from real.camera import Camera


class Robot(object):
    def __init__(self, is_sim, obj_mesh_dir, num_obj, workspace_limits,
                 tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                 is_testing, test_preset_cases, test_preset_file):

        self.is_sim = is_sim

        # If in simulation...
        if self.is_sim:
            pass
        # If in real-settings...
        else:
            # self.r = urx.Robot(tcp_host_ip)
            # self.r.set_tcp((0, 0, 0, 0, 0, 0))
            # self.r.set_payload(0.5, (0, 0, 0))

            # self.gripper = Robotiq_Two_Finger_Gripper(self.r)

            # NOTE: this is for D415
            # home_in_deg = np.array(
            # [-151.4, -93.7, 85.4, -90, -90, 0]) * 1.0

            # Default home joint configuration
            # NOTE: this is orig
            # self.home_joint_config = [-(180.0/360.0)*2*np.pi, -(84.2/360.0)*2*np.pi,
            # (112.8/360.0)*2*np.pi, -(119.7/360.0)*2*np.pi, -(90.0/360.0)*2*np.pi, 0.0]

            # NOTE this is only for calibrate.py (reduce retry time) - #
            # checkerboard flat and pointing up
            # self.home_joint_config = [-np.pi, -
            # np.pi/2, np.pi/2, 0, np.pi/2, np.pi]

            # Default joint speed configuration
            # self.joint_acc = 8 # Safe: 1.4
            # self.joint_vel = 3 # Safe: 1.05
            self.joint_acc = 1.0  # Safe when set 30% spe71ed on pendant
            self.joint_vel = 0.7

            # Default tool speed configuration
            # self.tool_acc = 1.2 # Safe: 0.5
            # self.tool_vel = 0.25 # Safe: 0.2
            # self.tool_acc = 0.1  # Safe when set 30% speed on pendant
            # self.tool_vtel = 0.1

            # Connect to robot client
            # self.tcp_host_ip = tcp_host_ip
            # self.tcp_port = tcp_port

            # port is assumed to be 30002
            self.r = URcomm(tcp_host_ip, self.joint_vel,
                            self.joint_acc)

            # Move robot to home pose
            self.r.go_home()
            self.r.activate_gripper()
            # self.r.close_gripper()
            # self.r.open_gripper()

            '''
            # Fetch RGB-D data from RealSense camera
            self.camera = Camera()
            self.cam_intrinsics = self.camera.intrinsics

            # Load camera pose (from running calibrate.py), intrinsics and depth scale
            self.cam_pose = np.loadtxt("real/camera_pose.txt", delimiter=" ")
            self.cam_depth_scale = np.loadtxt(
                "real/camera_depth_scale.txt", delimiter=" ")
            '''

    def get_camera_data(self):

        if self.is_sim:
            pass

        else:
            # Get color and depth image from ROS service
            color_img, depth_img = self.camera.get_data()
            # color_img = self.camera.color_data.copy()
            # depth_img = self.camera.depth_data.copy()

        return color_img, depth_img

    def grasp_object(self, position, orientation):
        # throttle z position
        # position[2] = max(position[2] - 0.050, self.workspace_limits[2][0])
        position[2] = max(position[2] - 0.050, self.moveto_limits[2][0])

        self.open_gripper()
        # move fast to right above the object
        # height of gripper?
        self.move_to([position[0], position[1], position[2] + 0.150],
                     orientation)
        # then slowly move down
        self.move_to(position, orientation,
                     acc_scaling=0.5, vel_scaling=0.1)
        # and grasp it
        self.close_gripper()

    def grasp(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: grasp at (%f, %f, %f)' %
              (position[0], position[1], position[2]))

        if self.is_sim:
            pass

            # Compute tool orientation from heightmap rotation angle
            # Basically, how should I rotate the gripper...
            # I GUESS THIS IS KIND OF IMPORTANT
            # It would be nice to specify in terms of ... pis, and not rx ry rz
        else:
            grasp_orientation = [1.0, 0.0]
            '''
            if heightmap_rotation_angle > np.pi:
                heightmap_rotation_angle = heightmap_rotation_angle - 2*np.pi
            tool_rotation_angle = heightmap_rotation_angle/2
            tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(
                tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
            tool_orientation_angle = np.linalg.norm(tool_orientation)
            tool_orientation_axis = tool_orientation/tool_orientation_angle
            tool_orientation_rotm = utils.angle2rotm(
                tool_orientation_angle, tool_orientation_axis, point=None)[:3, :3]

            # Compute tilted tool orientation during dropping into bin
            tilt_rotm = utils.euler2rotm(np.asarray([-np.pi/4, 0, 0]))
            tilted_tool_orientation_rotm = np.dot(
                tilt_rotm, tool_orientation_rotm)
            tilted_tool_orientation_axis_angle = utils.rotm2angle(
                tilted_tool_orientation_rotm)
            tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(
                tilted_tool_orientation_axis_angle[1:4])
            '''
            # position
            # TODO: FIX
            # tool_orientation = [2.21, 2.19, -0.04]
            tool_orientation = [2.22, -2.22, 0]
            tilted_tool_orientation = tool_orientation
            # Attempt grasp
            print('!--- Attempting to open gripper, then go down & close --!')
            self.grasp_object(position, tool_orientation)
            '''
            position = np.asarray(position).copy()
            position[2] = max(position[2] - 0.05, workspace_limits[2][0])

            tcp_command = "def process():\n"
            # ... is this a way to close the gripper
            # sure is
            tcp_command += " set_digital_out(8,False)\n"
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" \
                % (position[0], position[1], position[2] + 0.1, tool_orientation[0],
                   tool_orientation[1], 0.0, self.joint_acc * 0.5, self.joint_vel * 0.5)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" \
                % (position[0], position[1], position[2], tool_orientation[0],
                   tool_orientation[1], 0.0, self.joint_acc * 0.1, self.joint_vel * 0.1)
            tcp_command += " set_digital_out(8,True)\n"
            tcp_command += "end\n"
            '''
            # Block until robot reaches target tool position and gripper fingers have stopped moving
            '''
            # state_data = self.get_state()
            # tool_analog_input2 = self.parse_tcp_state_data(
            # state_data, 'tool_data')

            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))

            tool_analog_input2 = self.parse_tcp_state_data(
                self.tcp_socket, 'tool_data')
            timeout_t0 = time.time()

            while True:
                # state_data = self.get_state()
                self.tcp_socket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))

                new_tool_analog_input2 = self.parse_tcp_state_data(
                    self.tcp_socket, 'tool_data')
                print('tool analog input', new_tool_analog_input2)
                actual_tool_pose = self.parse_tcp_state_data(
                    self.tcp_socket, 'cartesian_info')
                timeout_t1 = time.time()
                self.tcp_socket.close()
            '''
            # TODO: determine if gripper has force on it
            '''
                if (tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - position[j]) < self.tool_pose_tolerance[j] for j in range(3)])) or (timeout_t1 - timeout_t0) > 5:
                    print('Breaking')
                    break
                tool_analog_input2 = new_tool_analog_input2
            '''

            # Check if gripper is open (grasp might be successful)
            # gripper_full_closed = self.close_gripper()
            # grasp_success = not gripper_full_closed
            # gripper_open = tool_analog_input2 > 0.26
            # gripper_open = !gripper_full_closed

            # # Check if grasp is successful
            # grasp_success =  tool_analog_input2 > 0.26

            # orig
            # home_position = [0.49, 0.11, 0.03]
            # bin_position = [0.5, -0.45, 0.1]

            # NOTE: mine
            bin_position = [0.580, -0.040, 0.300]
            # home_position = [0.400, 0.000, 0.260]
            # NOTE: mine, and doesn't block the view
            # home_position = [0.400, -0.100, 0.420]
            # D435 home_position = [0.254, 0.218, 0.434]
            # D415
            home_position = [0.360, 0.180, 0.504]
            home_orientation = [2.78, -1.67, 0.17]

            # If gripper is open, drop object in bin and check if grasp is successful
            # grasp_success = False
            # NOTE: last minute change (why keep grasping same spot)
            grasp_success = True

            # gripper_full_closed = self.check_grasp()
            gripper_full_closed = False
            # print('Gripper state', gripper_full_closed)
            if not gripper_full_closed:  # yay we might have grabbed something
                # Pre-compute blend radius
                # blend_radius = min(
                # abs(bin_position[1] - position[1])/2 - 0.01, 0.2)

                # Attempt placing in bin
                print("attempting to drop into bin and then go home")
                self.move_to(bin_position, None)
                self.open_gripper()

                print('Going home now')
                self.move_to(home_position, home_orientation)

                # NOTE: original code separates into approach and throw (tilted) parts
                '''
                self.tcp_socket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
                tcp_command = "def process():\n"
                tcp_command += "movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % \
                    (position[0], position[1], bin_position[2],
                     tool_orientation[0], tool_orientation[1], 0.0,
                     self.joint_acc, self.joint_vel, blend_radius)
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % \
                    (bin_position[0], bin_position[1], bin_position[2],
                     tilted_tool_orientation[0], tilted_tool_orientation[1],
                     tilted_tool_orientation[2], self.joint_acc, self.joint_vel,
                     blend_radius)
                tcp_command += " set_digital_out(8,False)\n"
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.0)\n" % \
                    (home_position[0], home_position[1], home_position[2],
                     tool_orientation[0], tool_orientation[1], 0.0,
                     self.joint_acc*0.5, self.joint_vel*0.5)
                tcp_command += "end\n"
                self.tcp_socket.send(str.encode(tcp_command))
                self.tcp_socket.close()
                '''
                # print(tcp_command) # Debug

                # Measure gripper width until robot reaches near bin location
                # state_data = self.get_state()
                measurements = []
                '''
                while True:
                    # state_data = self.get_state()

                    self.tcp_socket = socket.socket(
                        socket.AF_INET, socket.SOCK_STREAM)
                    self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
                    tool_analog_input2 = self.parse_tcp_state_data(
                        self.tcp_socket, 'tool_data')
                    actual_tool_pose = self.parse_tcp_state_data(
                        self.tcp_socket, 'cartesian_info')
                    self.tcp_socket.close()

                    measurements.append(tool_analog_input2)
                    if abs(actual_tool_pose[1] - bin_position[1]) < 0.2 or all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                        print('\n !------ Gripper not closed; breaking --')
                        break
                '''

                # TODO!
                # TODO: this appears to continuously try to close to keep object
                # in grasp (in case of slip when moving); mine just closes !
                # If gripper width did not change before reaching bin location, then object is in grip and grasp is successful
                if len(measurements) >= 2:
                    if abs(measurements[0] - measurements[1]) < 0.1:
                        print('\n !------ Grasp success, did not fall out!---')
                        grasp_success = True

            else:
                print('\n !------ Gripper closed ---')
                '''
                self.tcp_socket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
                tcp_command = "def process():\n"
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % \
                    (position[0], position[1], position[2]+0.1,
                     tool_orientation[0], tool_orientation[1], 0.0,
                     self.joint_acc*0.5, self.joint_vel*0.5)
                tcp_command += "movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.0)\n" %  \
                    (home_position[0], home_position[1], home_position[2],
                     tool_orientation[0], tool_orientation[1], 0.0,
                     self.joint_acc*0.5, self.joint_vel*0.5)
                tcp_command += "end\n"
                self.tcp_socket.send(str.encode(tcp_command))
                self.tcp_socket.close()
                '''

            print('\n !------ Gripper closed, process() defined ---')
            # Block until robot reaches home location
            # state_data = self.get_state()
            '''
            self.tcp_socket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tool_analog_input2 = self.parse_tcp_state_data(
                self.tcp_socket, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(
                self.tcp_socket, 'cartesian_info')
            self.tcp_socket.close()

            while True:
                # state_data = self.get_state()
                self.tcp_socket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
                new_tool_analog_input2 = self.parse_tcp_state_data(
                    self.tcp_socket, 'tool_data')
                actual_tool_pose = self.parse_tcp_state_data(
                    self.tcp_socket, 'cartesian_info')
                self.tcp_socket.close()

                if (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                    print('\n !------ Gripper closed; loop breaking ---')
                    break

                tool_analog_input2 = new_tool_analog_input2
            '''

        return grasp_success

    def restart_real(self):
        print('DEBUG: restarting real')


'''
def restart_real(self):
    # Compute tool orientation from heightmap rotation angle
    grasp_orientation = [1.0, 0.0]
    tool_rotation_angle = -np.pi/4
    tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(
        tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
    tool_orientation_angle = np.linalg.norm(tool_orientation)
    tool_orientation_axis = tool_orientation/tool_orientation_angle
    tool_orientation_rotm = utils.angle2rotm(
        tool_orientation_angle, tool_orientation_axis, point=None)[:3, :3]

    tilt_rotm = utils.euler2rotm(np.asarray([-np.pi/4, 0, 0]))
    tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
    tilted_tool_orientation_axis_angle = utils.rotm2angle(
        tilted_tool_orientation_rotm)
    tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(
        tilted_tool_orientation_axis_angle[1:4])

    # Move to box grabbing position
    box_grab_position = [0.5, -0.35, -0.12]
    self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
    tcp_command = "def process():\n"
    tcp_command += " set_digital_out(8,False)\n"
    tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0], box_grab_position[1], box_grab_position[2] +
                                                                            0.1, tilted_tool_orientation[0], tilted_tool_orientation[1], tilted_tool_orientation[2], self.joint_acc, self.joint_vel)
    tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0], box_grab_position[1],
                                                                            box_grab_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc, self.joint_vel)
    tcp_command += " set_digital_out(8,True)\n"
    tcp_command += "end\n"
    self.tcp_socket.send(str.encode(tcp_command))
    self.tcp_socket.close()

    # Block until robot reaches box grabbing position and gripper fingers have stopped moving
    state_data = self.get_state()
    tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
    while True:
        state_data = self.get_state()
        new_tool_analog_input2 = self.parse_tcp_state_data(
            state_data, 'tool_data')
        actual_tool_pose = self.parse_tcp_state_data(
            state_data, 'cartesian_info')
        if tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - box_grab_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            break
        tool_analog_input2 = new_tool_analog_input2

    # Move to box release position
    box_release_position = [0.5, 0.08, -0.12]
    home_position = [0.49, 0.11, 0.03]
    self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
    tcp_command = "def process():\n"
    tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0], box_release_position[1],
                                                                            box_release_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc*0.1, self.joint_vel*0.1)
    tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0], box_release_position[1],
                                                                            box_release_position[2]+0.3, tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc*0.02, self.joint_vel*0.02)
    tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.29)\n" % (box_grab_position[0]-0.05, box_grab_position[1]+0.1, box_grab_position[2] +
                                                                            0.3, tilted_tool_orientation[0], tilted_tool_orientation[1], tilted_tool_orientation[2], self.joint_acc*0.5, self.joint_vel*0.5)
    tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0]-0.05, box_grab_position[1]+0.1,
                                                                            box_grab_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc*0.5, self.joint_vel*0.5)
    tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0], box_grab_position[1],
                                                                            box_grab_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc*0.1, self.joint_vel*0.1)
    tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0]+0.05, box_grab_position[1],
                                                                            box_grab_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc*0.1, self.joint_vel*0.1)
    tcp_command += " set_digital_out(8,False)\n"
    tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0], box_grab_position[1], box_grab_position[2] +
                                                                            0.1, tilted_tool_orientation[0], tilted_tool_orientation[1], tilted_tool_orientation[2], self.joint_acc, self.joint_vel)
    tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (home_position[0], home_position[1],
                                                                            home_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc, self.joint_vel)
    tcp_command += "end\n"
    self.tcp_socket.send(str.encode(tcp_command))
    self.tcp_socket.close()

    # Block until robot reaches home position
    state_data = self.get_state()
    tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
    while True:
        state_data = self.get_state()
        new_tool_analog_input2 = self.parse_tcp_state_data(
            state_data, 'tool_data')
        actual_tool_pose = self.parse_tcp_state_data(
            state_data, 'cartesian_info')
        if tool_analog_input2 > 3.0 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            break
        tool_analog_input2 = new_tool_analog_input2
'''
