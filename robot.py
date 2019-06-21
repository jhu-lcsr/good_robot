import socket
import select
import struct
import time
import os
import numpy as np
import utils
import serial
import binascii

from simulation import vrep


class Robot(object):
    def __init__(self, is_sim, obj_mesh_dir, num_obj, workspace_limits,
                 tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                 is_testing, test_preset_cases, test_preset_file):

        self.is_sim = is_sim
        self.workspace_limits = workspace_limits

        # If in simulation...
        if self.is_sim:
            pass
        # If in real-settings...
        else:

            # Connect to robot client
            self.tcp_host_ip = tcp_host_ip
            self.tcp_port = tcp_port
            # self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Connect as real-time client to parse state data
            self.rtc_host_ip = rtc_host_ip
            self.rtc_port = rtc_port

            # Default home joint configuration
            # NOTE: this is for debug (hardcode calib) testing
            # self.home_joint_config = [-np.pi, -(80/360.) * 2 * np.pi, np.pi/2,
            # -np.pi/2, -np.pi/2, 0]

            # NOTE: This is home so arm does not block depth cam
            # home_in_deg = np.array([-191, -117, 116, -93, -91, -11]) * 1.0
            # NOTE: This is for main.py to unblock
            home_in_deg = np.array([-158, -114, 109, -85, -88, +20]) * 1.0
            self.home_joint_config = np.deg2rad(home_in_deg)

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
            self.joint_acc = 0.12  # Safe when set 30% speed on pendant
            self.joint_vel = 0.35

            # Joint tolerance for blocking calls
            self.joint_tolerance = 0.01

            # Default tool speed configuration
            # self.tool_acc = 1.2 # Safe: 0.5
            # self.tool_vel = 0.25 # Safe: 0.2
            self.tool_acc = 0.1  # Safe when set 30% speed on pendant
            self.tool_vel = 0.1

            # Tool pose tolerance for blocking calls
            self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

            # Move robot to home pose
            # TODO: activate gripper function
            self.activate_gripper()
            self.go_home()
            # time.sleep(1)
            # self.close_gripper()
            # self.open_gripper()

            # Fetch RGB-D data from RealSense camera
            # TODO Fix camera
            from real.camera import Camera
            self.camera = Camera()
            self.cam_intrinsics = self.camera.intrinsics

            # Load camera pose (from running calibrate.py), intrinsics and depth scale
            # NOTE: Is this independent of where the camera is?
            self.cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
            self.cam_depth_scale = np.loadtxt(
                'real/camera_depth_scale.txt', delimiter=' ')

    def reposition_objects(self, workspace_limits):

        # Move gripper out of the way
        self.move_to([-0.1, 0, 0.3], None)
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        # vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        # time.sleep(1)

        for object_handle in self.object_handles:

            # Drop object at random x,y location and random orientation in robot workspace
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * \
                np.random.random_sample() + workspace_limits[0][0] + 0.1
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * \
                np.random.random_sample() + workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi *
                                  np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            vrep.simxSetObjectPosition(
                self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(
                self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)
            time.sleep(2)

    def get_camera_data(self):

        if self.is_sim:
            pass

        else:
            # Get color and depth image from ROS service
            color_img, depth_img = self.camera.get_data()
            # color_img = self.camera.color_data.copy()
            # depth_img = self.camera.depth_data.copy()

        return color_img, depth_img

    # def parse_tcp_state_data(self, state_data, subpackage):
    def parse_tcp_state_data(self, tcp_socket, subpackage):
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect((self.tcp_host_ip, self.tcp_port))

        state_data = tcp_socket.recv(1024)
        state_data = tcp_socket.recv(1024)
        # Read package header

        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0]
        robot_message_type = data_bytes[4]
        # print('robot message type', robot_message_type)
        while (robot_message_type != 16):
            print('Wrong message type, trying again')
            state_data = tcp_socket.recv(1024)
            state_data = tcp_socket.recv(2048)
            data_bytes = bytearray()
            data_bytes.extend(state_data)
            data_length = struct.unpack("!i", data_bytes[0:4])[0]
            robot_message_type = data_bytes[4]
            print('robot message type', robot_message_type)

        byte_idx = 5

        # Parse sub-packages
        subpackage_types = {
            'joint_data': 1, 'cartesian_info': 4, 'force_mode_data': 7, 'tool_data': 2}
        while byte_idx < data_length:
            # package_length = int.from_bytes(data_bytes[byte_idx:(byte_idx+4)], byteorder='big', signed=False)
            package_length = struct.unpack(
                "!i", data_bytes[byte_idx:(byte_idx+4)])[0]
            byte_idx += 4
            package_idx = data_bytes[byte_idx]
            if package_idx == subpackage_types[subpackage]:
                byte_idx += 1
                break
            byte_idx += package_length - 4
        tcp_socket.close()

        def parse_joint_data(data_bytes, byte_idx):
            actual_joint_positions = [0, 0, 0, 0, 0, 0]
            target_joint_positions = [0, 0, 0, 0, 0, 0]
            for joint_idx in range(6):
                actual_joint_positions[joint_idx] = struct.unpack(
                    '!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
                target_joint_positions[joint_idx] = struct.unpack(
                    '!d', data_bytes[(byte_idx+8):(byte_idx+16)])[0]
                byte_idx += 41
            # DEBUG:
            # print('joint pos', actual_joint_positions)
            return actual_joint_positions

        def parse_cartesian_info(data_bytes, byte_idx):
            actual_tool_pose = [0, 0, 0, 0, 0, 0]
            for pose_value_idx in range(6):
                actual_tool_pose[pose_value_idx] = struct.unpack(
                    '!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
                byte_idx += 8
            # print('tool pos', actual_tool_pose)
            return actual_tool_pose

        def parse_tool_data(data_bytes, byte_idx):
            byte_idx += 2
            tool_analog_input2 = struct.unpack(
                '!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            return tool_analog_input2

        parse_functions = {'joint_data': parse_joint_data,
                           'cartesian_info': parse_cartesian_info, 'tool_data': parse_tool_data}
        return parse_functions[subpackage](data_bytes, byte_idx)

    def parse_rtc_state_data(self, state_data):

        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0]
        assert(data_length == 812)
        byte_idx = 4 + 8 + 8*48 + 24 + 120
        TCP_forces = [0, 0, 0, 0, 0, 0]
        for joint_idx in range(6):
            TCP_forces[joint_idx] = struct.unpack(
                '!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            byte_idx += 8

        return TCP_forces

    def activate_gripper(self, async=False):
        print('!-- activating gripper')
        ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=1,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            bytesize=serial.EIGHTBITS)
        # closing_force = '\xFF' # 255
        clear_rACT = "\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30"
        ser.write(clear_rACT)
        set_rACT = "\x09\x10\x03\xE8\x00\x03\x06\x01\x00\x00\x00\x00\x00\x72\xE1"
        ser.write(set_rACT)

        # read gripper status until activation completed?
        ser.write("\x09\x03\x07\xD0\x00\x01\x85\xCF")
        ser.readline()

        ser.close()

    def close_gripper(self, async=False):
        print("!-- close gripper")

        if self.is_sim:
            pass

        else:
            # NOTE: Adapted to Robotiq
            ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=1,
                                parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE,
                                bytesize=serial.EIGHTBITS)
            ser.write(
                "\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x42\x29")

            """
            # NOTE: THIS DOESN'T WORK (probably some hex vs str thing)
            closing_force = '\xBB'  # 187
            # close_cmd = "\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF" + \
                # closing_force + "\x42\x29"
            # print('close_cmd', close_cmd)
            # ser.write(close_cmd)
            """
            ser.close()
            if async:
                gripper_fully_closed = True
            else:
                time.sleep(0.2)
                gripper_fully_closed = self.check_grasp()

            # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            # tcp_command = "set_digital_out(8,True)\n"
            # self.tcp_socket.send(str.encode(tcp_command))
            # self.tcp_socket.close()

            return gripper_fully_closed

    # DONE convert to robotiq
    def open_gripper(self, async=False):

        if self.is_sim:
            pass
        else:
            print('!-- opening gripper')
            # NOTE: for robotiq gripper
            ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200,
                                timeout=1, parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE,
                                bytesize=serial.EIGHTBITS)
            ser.write(
                "\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19")
            # NOTE: Originally for RG2
            # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            # tcp_command = "set_digital_out(8,False)\n"
            # self.tcp_socket.send(str.encode(tcp_command))
            # self.tcp_socket.close()
            ser.close()
            if not async:
                time.sleep(0.2)

    def get_state(self):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(2048)
        state_data = self.tcp_socket.recv(2048)
        self.tcp_socket.close()
        return state_data

    def btw(self, a, min, max):
        if (a >= min) and (a <= max):
            return True
        return False

    def move_to(self, tool_position, tool_orientation, acc_scaling=1,
                vel_scaling=1):
        acc, vel = self.joint_acc * acc_scaling, self.joint_vel * vel_scaling

        if self.is_sim:
            pass
        limits = self.workspace_limits
        # is_safe = False
        if self.btw(tool_position[0], limits[0][0], limits[0][1]) and \
                self.btw(tool_position[1], limits[1][0], limits[1][1]) and \
                self.btw(tool_position[2], limits[2][0], limits[2][1]):
            # print("I guess it's safe")
            # print('DEBUG: Entered move_to function, going to ', tool_position,
            # tool_orientation)
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))

            if tool_orientation == None:
                curr_pose = self.parse_tcp_state_data(self.tcp_socket,
                                                      'cartesian_info')
                print('DEBUG: Attempting to only move position')
                # NOTE: I changed to movej
                tcp_command = "movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" %  \
                    (tool_position[0], tool_position[1], tool_position[2],
                     curr_pose[3], curr_pose[4], curr_pose[5], acc, vel)
                # print('DEBUG: tcp command', tcp_command)

            else:
                tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % \
                    (tool_position[0], tool_position[1], tool_position[2],
                     tool_orientation[0], tool_orientation[1], tool_orientation[2],
                     acc, vel)
                print('move_to', tool_position, tool_orientation)
            self.tcp_socket.send(str.encode(tcp_command))

            # Block until robot reaches target tool position
            #
            # TODO figure out why, have to run twice tcp_state_data =
            # self.tcp_socket.recv(2048) tcp_state_data =
            # self.tcp_socket.recv(2048)
            actual_tool_pose = self.parse_tcp_state_data(self.tcp_socket,
                                                         'cartesian_info')
            while not all([np.abs(actual_tool_pose[j] - tool_position[j]) <
                           self.tool_pose_tolerance[j] for j in range(3)]):
                # print('DEBUG: MoveL!, but not quite there yet, hold on...')
                # tcp_state_data = self.tcp_socket.recv(2048)
                prev_actual_tool_pose = np.asarray(actual_tool_pose).copy()
                actual_tool_pose = self.parse_tcp_state_data(self.tcp_socket,
                                                             'cartesian_info')
                time.sleep(0.01)
                self.tcp_socket.close()
        else:
            print("DEBUG: It's not safe to move here!", tool_position, limits)

    """
    def guarded_move_to(self, tool_position, tool_orientation):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))

        # Read actual tool position
        tcp_state_data = self.tcp_socket.recv(2048)
        # TODO this will break if run
        actual_tool_pose = self.parse_tcp_state_data(
            tcp_state_data, 'cartesian_info')
        execute_success = True

        # Increment every cm, check force
        self.tool_acc = 0.1  # 1.2 # 0.5

        while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]

            # Compute motion trajectory in 1cm increments
            increment = np.asarray(
                [(tool_position[j] - actual_tool_pose[j]) for j in range(3)])
            if np.linalg.norm(increment) < 0.01:
                increment_position = tool_position
            else:
                increment = 0.01*increment/np.linalg.norm(increment)
                increment_position = np.asarray(
                    actual_tool_pose[0:3]) + increment

            # Move to next increment position (blocking call)
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (increment_position[0], increment_position[1],
                                                                               increment_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.tool_acc, self.tool_vel)
            self.tcp_socket.send(str.encode(tcp_command))

            time_start = time.time()
            tcp_state_data = self.tcp_socket.recv(2048)
            tcp_state_data = self.tcp_socket.recv(2048)
            actual_tool_pose = self.parse_tcp_state_data(
                tcp_state_data, 'cartesian_info')
            while not all([np.abs(actual_tool_pose[j] - increment_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                # print([np.abs(actual_tool_pose[j] - increment_position[j]) for j in range(3)])
                tcp_state_data = self.tcp_socket.recv(2048)
                tcp_state_data = self.tcp_socket.recv(2048)
                actual_tool_pose = self.parse_tcp_state_data(
                    tcp_state_data, 'cartesian_info')
                time_snapshot = time.time()
                if time_snapshot - time_start > 1:
                    break
                time.sleep(0.01)

            # Reading TCP forces from real-time client connection
            rtc_state_data = self.rtc_socket.recv(6496)
            TCP_forces = self.parse_rtc_state_data(rtc_state_data)

            # If TCP forces in x/y exceed 20 Newtons, stop moving
            # print(TCP_forces[0:3])
            if np.linalg.norm(np.asarray(TCP_forces[0:2])) > 20 or (time_snapshot - time_start) > 1:
                print('Warning: contact detected! Movement halted. TCP forces: [%f, %f, %f]' % (
                    TCP_forces[0], TCP_forces[1], TCP_forces[2]))
                execute_success = False
                break

            time.sleep(0.01)

        self.tool_acc = 1.2  # 1.2 # 0.5

        self.tcp_socket.close()
        self.rtc_socket.close()

        return execute_success
    """

    def move_joints(self, joint_configuration):
        # DEBUG:
        # print('Entered move_joints function')
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movej([%f" % joint_configuration[0]  # NOTE: no p
        for joint_idx in range(1, 6):
            tcp_command = tcp_command + \
                (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + \
            "],a=%f,v=%f)\n" % (self.joint_acc, self.joint_vel)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        # state_data = self.tcp_socket.recv(2048)  # TODO why need to run twice
        # state_data = self.tcp_socket.recv(2048)
        actual_joint_positions = self.parse_tcp_state_data(
            self.tcp_socket, 'joint_data')
        while not all([np.abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
            # state_data = self.tcp_socket.recv(2048)
            # state_data = self.tcp_socket.recv(4096)
            actual_joint_positions = self.parse_tcp_state_data(
                self.tcp_socket, 'joint_data')
            time.sleep(0.01)
            # DEBUG:
            # print('MoveJ, but not quite there yet, hold on...')

        self.tcp_socket.close()

    def go_home(self):

        print('Going home!')
        self.move_joints(self.home_joint_config)

    # Note: must be preceded by close_gripper()
    def check_grasp(self):

        ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200,
                            timeout=1, parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            bytesize=serial.EIGHTBITS)
        ser.write(
            "\x09\x03\x07\xD0\x00\x03\x04\x0E")
        data_raw = ser.readline()
        data = binascii.hexlify(data_raw)
        position = int(data[14:16], 16)  # hex to dec
        ser.close()
        print('Position', position, ' is grasp closed? ', position > 215)
        return position > 215  # 230 is closed

        # Note: Original
        # state_data = self.get_state()
        # tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        # return tool_analog_input2 > 0.26

    # Primitives ----------------------------------------------------------

    # TODO probably need to change bin and home positions

    def grasp_object(self, position, orientation):
        # throttle z position
        position[2] = max(position[2] - 0.050, self.workspace_limits[2][0])

        self.open_gripper()
        # move fast to right above the object
        self.move_to([position[0], position[1], position[2] + 0.100],
                     orientation)
        # then slowly move down
        self.move_to(position, orientation, acc_scaling=0.5, vel_scaling=0.1)
        # and grasp it
        self.close_gripper()

    def grasp(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: grasp at (%f, %f, %f)' %
              (position[0], position[1], position[2]))

        if self.is_sim:
            pass

        else:

            # Compute tool orientation from heightmap rotation angle
            # Basically, how should I rotate the gripper...
            # I GUESS THIS IS KIND OF IMPORTANT
            # It would be nice to specify in terms of ... pis, and not rx ry rz
            grasp_orientation = [1.0, 0.0]
            """
            if heightmap_rotation_angle > np.pi:
                heightmap_rotation_angle = heightmap_rotation_angle - 2*np.pi
            tool_rotation_angle = heightmap_rotation_angle/2
            tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(
                tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
            tool_orientation_angle = np.linalg.norm(tool_orientation)
            tool_orientation_axis = tool_orientation/tool_orientation_angle
            tool_orientation_rotm = utils.angle2rotm(
                tool_orientation_angle, tool_orientation_axis, point=None)[:3, :3]
            """

            # Compute tilted tool orientation during dropping into bin
            """
            tilt_rotm = utils.euler2rotm(np.asarray([-np.pi/4, 0, 0]))
            tilted_tool_orientation_rotm = np.dot(
                tilt_rotm, tool_orientation_rotm)
            tilted_tool_orientation_axis_angle = utils.rotm2angle(
                tilted_tool_orientation_rotm)
            tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(
                tilted_tool_orientation_axis_angle[1:4])
            """
            # position
            # TODO: FIX
            # tool_orientation = [2.21, 2.19, -0.04]
            tool_orientation = [2.22, -2.22, 0]
            tilted_tool_orientation = tool_orientation

            # Attempt grasp

            print('!--- Attempting to open gripper, then go down & close --!')
            self.grasp_object(position, tool_orientation)
            """
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
            """
            # Block until robot reaches target tool position and gripper fingers have stopped moving
            """
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
            """

            # TODO: determine if gripper has force on it
            """
                if (tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - position[j]) < self.tool_pose_tolerance[j] for j in range(3)])) or (timeout_t1 - timeout_t0) > 5:
                    print('Breaking')
                    break
                tool_analog_input2 = new_tool_analog_input2
                """

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
            home_position = [0.254, 0.218, 0.434]

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
                self.move_to(home_position, None)

                # NOTE: original code separates into approach and throw (tilted) parts
                """
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
                """
                # print(tcp_command) # Debug

                # Measure gripper width until robot reaches near bin location
                # state_data = self.get_state()
                measurements = []
                """
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
                    """

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

                """
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
                """
                print('\n !------ Gripper closed, process() defined ---')

            # Block until robot reaches home location
            # state_data = self.get_state()

            """
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
                """

        return grasp_success

    def restart_real(self):
        print('DEBUG: restarting real')


"""
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
        """

# def place(self, position, orientation, workspace_limits):
#     print('Executing: place at (%f, %f, %f)' % (position[0], position[1], position[2]))

#     # Attempt placing
#     position[2] = max(position[2], workspace_limits[2][0])
#     self.move_to([position[0], position[1], position[2] + 0.2], orientation)
#     self.move_to([position[0], position[1], position[2] + 0.05], orientation)
#     self.tool_acc = 1 # 0.05
#     self.tool_vel = 0.02 # 0.02
#     self.move_to([position[0], position[1], position[2]], orientation)
#     self.open_gripper()
#     self.tool_acc = 1 # 0.5
#     self.tool_vel = 0.2 # 0.2
#     self.move_to([position[0], position[1], position[2] + 0.2], orientation)
#     self.close_gripper()
#     self.go_home()

# def place(self, position, heightmap_rotation_angle, workspace_limits):
#     print('Executing: place at (%f, %f, %f)' % (position[0], position[1], position[2]))

#     if self.is_sim:

#         # Approach place target
#         self.move_to(position, None)

#         # Ensure gripper is open
#         self.open_gripper()

#         # Move gripper to location above place target
#         self.move_to(location_above_place_target, None)

#         place_success = True
#         return place_success
