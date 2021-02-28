import socket
import select
import struct
import time
import os
import numpy as np
import itertools
import utils
import traceback
import copy
from simulation import vrep
from scipy import ndimage, misc
from glob import glob
try:
    from gripper.robotiq_2f_gripper_ctrl import RobotiqCGripper
except ImportError:
    print('Real robotiq gripper control is not available. '
          'Ensure pymodbus is installed:\n'
          '    pip3 install --user --upgrade pymodbus\n')
    RobotiqCGripper = None

def gripper_control_pose_to_arm_control_pose(gripper_translation, gripper_orientation, gripper_to_arm_transform=None):
        # arm_trans = np.eye(4,4)
        # arm_trans[0:3,3] = np.asarray(gripper_translation)
        # # gripper_orientation = [-gripper_orientation[0], -gripper_orientation[1], -gripper_orientation[2]]
        # arm_rotm = np.eye(4,4)
        # arm_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(gripper_orientation))
        # gripper_pose = np.dot(arm_trans, arm_rotm) # Compute rigid transformation representating camera pose
        if gripper_to_arm_transform is None:
            return gripper_translation, gripper_orientation
        gripper_pose = utils.axis_angle_and_translation_to_rigid_transformation(gripper_translation, gripper_orientation)
        # print('gripper_control_pose_to_arm_control_pose() gripper_pose: \n' + str(gripper_pose))
        # 4x4 transform of the arm pose
        arm_pose = np.dot(gripper_pose, utils.pose_inv(gripper_to_arm_transform))
        # arm_pose = np.dot(gripper_pose, gripper_to_arm_transform)
        # arm_pose = np.dot(utils.pose_inv(gripper_to_arm_transform), gripper_pose)
        # arm_pose = np.dot(gripper_to_arm_transform, gripper_pose)
        arm_orientation_axis_angle = utils.rotm2angle(arm_pose[0:3,0:3])
        arm_orientation = arm_orientation_axis_angle[0]*np.asarray(arm_orientation_axis_angle[1:4])
        arm_translation = arm_pose[0:3,3]
        return arm_translation, arm_orientation

def orientation_and_angle_to_push_direction(heightmap_rotation_angle, push_orientation=None):
    if push_orientation is None:
        push_orientation = [1.0, 0.0]
    # Compute push direction and endpoint (push to right of rotated heightmap)
    push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle), 0.0])
    return push_direction

def push_poses(heightmap_rotation_angle, position, workspace_limits, push_orientation=None, push_length=0.1, up_length=0.2, tilt_axis=None, gripper_to_arm_transform=None, buffer=0.005):
    """
    # Returns

    position, up_pos, push_endpoint, push_direction, tool_orientation_rotm
    """
    if push_orientation is None:
        push_orientation = [1.0, 0.0]
    if tilt_axis is None:
        tilt_axis = np.asarray([0,0,np.pi/2])
    # Compute push direction and endpoint (push to right of rotated heightmap)
    # Compute target location (push to the right)
    push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle), 0.0])
    target_x = min(max(position[0] + push_direction[0]*push_length, workspace_limits[0][0]), workspace_limits[0][1])
    target_y = min(max(position[1] + push_direction[1]*push_length, workspace_limits[1][0]), workspace_limits[1][1])
    push_endpoint = np.asarray([target_x, target_y, position[2] + buffer])
    push_direction.shape = (3, 1)

    # Compute tool orientation from heightmap rotation angle
    # TODO(ahundt) tool_rotation_angle, particularly dividing by 2, may affect (1) sim to real transfer, and (2) common sense checks, especially considering that our real robot gripper is not centered on the tool control point. Verify transforms!
    tool_rotation_angle = heightmap_rotation_angle/2
    # tool_orientation = orientation_and_angle_to_push_direction(tool_rotation_angle, push_orientation)*np.pi
    tool_orientation = np.asarray([push_orientation[0]*np.cos(tool_rotation_angle) - push_orientation[1]*np.sin(tool_rotation_angle), push_orientation[0]*np.sin(tool_rotation_angle) + push_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi

    tool_orientation_angle = np.linalg.norm(tool_orientation)
    tool_orientation_axis = tool_orientation/tool_orientation_angle
    tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]

    # Compute tilted tool orientation during push
    tilt_axis = np.dot(utils.euler2rotm(tilt_axis)[:3,:3], push_direction)
    tilt_rotm = utils.angle2rotm(-np.pi/8, tilt_axis, point=None)[:3,:3]
    tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
    tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
    tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])

    # Push only within workspace limits
    position = np.asarray(position).copy()
    position[0] = min(max(position[0], workspace_limits[0][0]), workspace_limits[0][1])
    position[1] = min(max(position[1], workspace_limits[1][0]), workspace_limits[1][1])
    position[2] = max(position[2] + buffer, workspace_limits[2][0] + buffer) # Add buffer to surface
    up_pos = np.array([position[0],position[1],position[2] + up_length])

    # convert to real arm tool control points, rather than heightmap based control points
    position, tool_orientation = gripper_control_pose_to_arm_control_pose(position, tool_orientation, gripper_to_arm_transform)
    up_pos, _ = gripper_control_pose_to_arm_control_pose(up_pos, tool_orientation, gripper_to_arm_transform)
    push_endpoint, tilted_tool_orientation = gripper_control_pose_to_arm_control_pose(push_endpoint, tilted_tool_orientation, gripper_to_arm_transform)

    return position, up_pos, push_endpoint, push_direction, tool_orientation, tilted_tool_orientation

class Robot(object):
    """
    Key member variables:
       self.color_space: list of colors to give objects
       self.color_names: list of strings identifying colors in color_space
       self.stored_action_labels: name of actions for stacking one hot encoding, set if self.grasp_color_task = True.
       self.object_handles: list of vrep object handles, the unique identifier integer indices needed for controlling objects in the vrep simulator
       self.workspace_limits: bounds of the valid object workspace.
    """
    def __init__(self, is_sim=True, obj_mesh_dir=None, num_obj=None, workspace_limits=None,
                 tcp_host_ip='192.168.1.155', tcp_port=502, rtc_host_ip=None, rtc_port=None,
                 is_testing=False, test_preset_cases=None, test_preset_file=None, test_preset_arr=None,
                 place=False, grasp_color_task=False, real_gripper_ip='192.168.1.11', calibrate=False,
                 unstack=False, heightmap_resolution=0.002, capture_logoblock_dataset=False, obj_scale=1,
                 textured=False, task_type=None):
        '''

        real_gripper_ip: None to assume the gripper is connected via the UR5,
             specify an ip address to directly use TCPModbus to talk directly with the gripper.
             Default is 192.168.1.11.
        '''
        self.is_sim = is_sim
        if workspace_limits is None:
            # workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
            if is_sim:
                workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.5]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
            else:
                # Corner near window on robot base side
                # [0.47984089 0.34192974 0.02173636]
                # Corner on the side of the cameras and far from the window
                # [ 0.73409861 -0.45199446 -0.00229499]
                # Dimensions of workspace should be 448 mm x 448 mm. That's 224x224 pixels with each pixel being 2mm x2mm.
                workspace_limits = np.asarray([[0.376, 0.824], [-0.264, 0.184], [-0.07, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        self.workspace_limits = workspace_limits
        self.heightmap_resolution = heightmap_resolution
        self.place_task = place
        self.unstack = unstack
        self.place_pose_history_limit = 6
        self.grasp_color_task = grasp_color_task
        self.sim_home_position = [-0.3, 0.0, 0.45]  # old value [-0.3, 0, 0.3]
        # self.gripper_ee_offset = 0.17
        # self.gripper_ee_offset = 0.15
        self.background_heightmap = None
        self.tool_tip_to_gripper_center_transform = None

        # list of place position attempts
        self.place_pose_history = []

        # HK: If grasping specific block color...
        #
        # TODO: Change to random color not just red block using  (b = [0, 1, 2, 3] np.random.shuffle(b)))
        # after grasping, put the block back
        self.color_names = ['blue', 'green', 'yellow', 'red', 'brown', 'orange', 'gray', 'purple', 'cyan', 'pink']

        # task type (defaults to None)
        self.task_type = task_type
        self.capture_logoblock_dataset = capture_logoblock_dataset

        # If in simulation...
        if self.is_sim:
            # Tool pose tolerance for blocking calls, [x, y, z, roll, pitch, yaw]
            # with units [m, m, m, rad, rad, rad]
            # TODO(ahundt) double check rad rad rad, it might be axis/angle where magnitude is rotation length.
            self.tool_pose_tolerance = [0.001,0.001,0.001,0.01,0.01,0.01]
            self.push_vertical_offset = 0.026
            if num_obj is None:
                num_obj = 10
            if obj_mesh_dir is None:
                obj_mesh_dir = os.path.abspath('objects/toys')
            # Define colors for object meshes (Tableau palette)
            # self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
            #                                [89.0, 161.0, 79.0], # green
            #                                [156, 117, 95], # brown
            #                                [242, 142, 43], # orange
            #                                [237.0, 201.0, 72.0], # yellow
            #                                [186, 176, 172], # gray
            #                                [255.0, 87.0, 89.0], # red
            #                                [176, 122, 161], # purple
            #                                [118, 183, 178], # cyan
            #                                [255, 157, 167]])/255.0 #pink
            self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                [89.0, 161.0, 79.0], # green
                                [237.0, 201.0, 72.0], # yellow
                                [255.0, 87.0, 89.0], # red
                                [156, 117, 95], # brown
                                [242, 142, 43], # orange
                                [186, 176, 172], # gray
                                [176, 122, 161], # purple
                                [118, 183, 178], # cyan
                                [255, 157, 167]])/255.0 #pink

            # Read files in object mesh directory
            self.obj_mesh_dir = obj_mesh_dir
            self.num_obj = num_obj
            # TODO(HK) specify which objects to load here from a command line parameter, should be able ot load one repeatedly
            #self.mesh_list = os.listdir(self.obj_mesh_dir)
            # Restrict only the .obj files 
            self.mesh_list = sorted(glob(os.path.join(self.obj_mesh_dir, "*.obj")))
            #print(f"self.meshlist: {self.mesh_list}")

            # Randomly choose objects to add to scene
            self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
            self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]

            # Make sure to have the server side running in V-REP:
            # in a child script of a V-REP scene, add following command
            # to be executed just once, at simulation start:
            #
            # simExtRemoteApiStart(19999)
            #
            # then start simulation, and run this program.
            #
            # IMPORTANT: for each successful call to simxStart, there
            # should be a corresponding call to simxFinish at the end!

            # MODIFY remoteApiConnections.txt

            if tcp_port is None or tcp_port == 30002 or tcp_port == 502:
                print("WARNING: default tcp port changed to 19997 for is_sim")
                tcp_port = 19997

            self.tcp_port = tcp_port
            #print(f"bp 1")
            self.restart_sim(connect=True)
            # initialize some startup state values and handles for
            # the joint configurations and home position
            # sim_joint_handles are unique identifying integers to control the robot joints in the simulator
            self.sim_joint_handles = []
            # home_joint_config will contain the floating point joint angles representing the home configuration.
            self.home_joint_config = []
            self.go_home()
            # set the home joint config based on the initialized simulation
            self.home_joint_config = self.get_joint_position()
            #print(f"got joints")

            self.is_testing = is_testing
            self.test_preset_cases = test_preset_cases
            self.test_preset_file = test_preset_file
            self.test_preset_arr = test_preset_arr

            # Setup virtual camera in simulation
            self.setup_sim_camera()

            # Scaling used when importing objects
            self.obj_scale = obj_scale
            self.textured = textured

            # If testing, read object meshes and poses from test case file
            print(f"self.is_testing {is_testing} self.test_preset_cases {self.test_preset_cases}")
            if self.is_testing and self.test_preset_cases:
                print(f"loading preset case")
                self.load_preset_case()

            # Add objects to simulation environment
            self.add_objects()

        # If in real-settings...
        else:
            # Tool pose tolerance for blocking calls, [x, y, z, roll, pitch, yaw]
            # with units [m, m, m, rad, rad, rad]
            # TODO(ahundt) double check rad rad rad, it might be axis/angle where magnitude is rotation length.
            self.tool_pose_tolerance = [0.002,0.002,0.002,0.01,0.01,0.01]
            self.push_vertical_offset = 0.01

            # Connect to robot client
            self.tcp_host_ip = tcp_host_ip
            self.tcp_port = tcp_port
            # self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Connect as real-time client to parse state data
            self.rtc_host_ip = rtc_host_ip
            self.rtc_port = rtc_port

            # Default home joint configuration
            # self.home_joint_config = [-np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
            # self.home_joint_config = [-(180.0/360.0)*2*np.pi, -(84.2/360.0)*2*np.pi, (112.8/360.0)*2*np.pi, -(119.7/360.0)*2*np.pi, -(90.0/360.0)*2*np.pi, 0.0]
            # Costar dataset home joint config
            # self.home_joint_config = [-0.202, -0.980, -1.800, -0.278, 1.460, 1.613]
            # Real Good Robot Home Joint Config (gripper low and near base)
            self.home_joint_config = [-0.021765167640454663, -0.7721485323791424, -2.137509664960675, -1.8396634790764201, 1.5608016750418263, 1.7122485182908058]
            self.home_cart = [0.16452807896085456, -0.1140799890027773, 0.3401360989767276, -0.25284986938091303, -3.0949552373620137, 0.018920323919325615]
            self.home_cart_low = self.home_cart
            # gripper high above scene and camera
            # self.home_joint_config = [0.2, -1.62, -0.85, -2.22, 1.57, 1.71]
            # self.home_cart = [0.4387869054651441, -0.022525365646335706, 0.6275609068446096, -0.09490323444344208, 3.1179780725241626, 0.004632836623511681]
            # self.home_cart_low = [0.4387869054651441, -0.022525365646335706, 0.3275609068446096, -0.09490323444344208, 3.1179780725241626, 0.004632836623511681]


            # Default joint speed configuration
            self.joint_acc = 4.0 # Safe: 1.4  Fast: 8
            self.joint_vel = 2.0 # Safe: 1.05  Fast: 3

            # Joint tolerance for blocking calls
            self.joint_tolerance = 0.01

            # Default tool speed configuration
            self.tool_acc = 1.0 # Safe: 0.5 Fast: 1.2
            self.tool_vel = 0.5 # Safe: 0.2 Fast: 0.5
            self.move_sleep = 1.0 # Safe: 2.0 Fast: 1.0

            # Initialize the real gripper based on user configuration
            if real_gripper_ip is None:
                self.gripper = None
            elif RobotiqCGripper is None:
                # Install instructions have already printed (see the imports section)
                # and we cannot run in this mode without pymodbus, so exit.
                exit(1)
            else:
                self.gripper = RobotiqCGripper(real_gripper_ip)
                self.gripper.wait_for_connection()

                self.gripper.reset()
                self.gripper.activate()

            # Move robot to home pose
            self.close_gripper()
            self.go_home()
            self.open_gripper()

            # Fetch RGB-D data from RealSense camera
            from real.camera import Camera
            self.camera = Camera(calibrate=calibrate)
            self.cam_intrinsics = self.camera.intrinsics

            # Load camera pose (from running calibrate.py), intrinsics and depth scale
            if not calibrate and os.path.isfile('real/robot_base_to_camera_pose.txt') and os.path.isfile('real/camera_depth_scale.txt'):
                self.cam_pose = np.loadtxt('real/robot_base_to_camera_pose.txt', delimiter=' ')
                self.cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')
            else:
                print('WARNING: Camera Calibration is not yet available, running calibrate_ros.py '
                      'will create the required files: real/robot_base_to_camera_pose.txt and real/camera_depth_scale.txt')
                # Camera calibration
                self.cam_pose = None
                self.cam_depth_scale = None

            # Get the transform to the gripper center, this is necessary when the robot control
            # poses differs from where the gripper center is, so a transform applying a correction is needed.
            if not calibrate and os.path.isfile('real/tool_tip_to_ar_tag_transform.txt'):
                self.tool_tip_to_gripper_center_transform = np.loadtxt('real/tool_tip_to_ar_tag_transform.txt', delimiter=' ')

            if os.path.isfile('real/background_heightmap.depth.png'):
                import cv2
                 # load depth image saved in 1e-5 meter increments
                 # see logger.py save_heightmaps() and trainer.py load_sample()
                 # for the corresponding save and load functions
                self.background_heightmap = np.array(cv2.imread('real/background_heightmap.depth.png', cv2.IMREAD_ANYDEPTH)).astype(np.float32) / 100000
                # TODO(ahundt) HACK REMOVE background_heightmap subtraction, COLLECT HEIGHTMAP AGAIN, SEE README.md for instructions
                # self.background_heightmap -= 0.03

            # real robot must use unstacking
            if self.place_task:
                self.unstack = True


    def load_preset_case(self, test_preset_file=None):
        if test_preset_file is None:
            #print(f"preset file is {self.test_preset_file}") 
            test_preset_file = self.test_preset_file
        file = open(test_preset_file, 'r')
        file_content = file.readlines()
        self.test_obj_mesh_files = []
        self.test_obj_mesh_colors = []
        self.test_obj_positions = []
        self.test_obj_orientations = []
        for object_idx in range(self.num_obj):
            file_content_curr_object = file_content[object_idx].split()
            self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir,file_content_curr_object[0]))
            self.test_obj_mesh_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
            self.test_obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
            self.test_obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
        file.close()

        self.obj_mesh_color = np.asarray(self.test_obj_mesh_colors)


    def setup_sim_camera(self):
        #print(f"getting camera data") 
        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)

        #print(f"camera bp 1") 
        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        #print(f"camera bp 2") 
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        #print(f"camera bp 3") 
        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

        #print(f"camera bp 4") 

    def generate_random_object_pose(self):
        drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
        drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
        object_position = [drop_x, drop_y, 0.15]
        object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
        return drop_x, drop_y, object_position, object_orientation


    def reposition_object_randomly(self, object_handle):
        """ randomly set a specific object's position and orientation on
        """
        drop_x, drop_y, object_position, object_orientation = self.generate_random_object_pose()
        # Drop object at random x,y location and random orientation in robot workspace
        vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)

        return object_position, object_orientation

    def reposition_object_at_list_index_randomly(self, list_index):
        object_handle = self.object_handles[list_index]
        self.reposition_object_randomly(object_handle)


    def reposition_object_at_list_index_to_location(self, obj_pos, obj_ori, index):
        """ Reposition the object to a specified position and orientation """
        object_handle = self.object_handles[index]
        # TODO(adit98) figure out significance of plane_handle, set to -1 for now
        if self.task_type is not None and self.task_type == 'unstack':
            plane_handle=-1
        else:
            success, plane_handle = vrep.simxGetObjectHandle(self.sim_client, "Plane", vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, object_handle, plane_handle, obj_pos, vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, object_handle, plane_handle, obj_ori, vrep.simx_opmode_blocking)


    def add_objects(self):

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        # object_handles is the list of unique vrep object integer identifiers, and it is how you control objects with the vrep API.
        # We need to keep track of the object names and the corresponding colors.
        self.object_handles = []
        self.vrep_names = []
        self.object_colors = []
        add_success = False
        failure_count = 0
        while not add_success:
            if (failure_count > 10 and failure_count %3 == 2) or len(self.object_handles) > len(self.obj_mesh_ind):
                # If the simulation is not currently running, attempt to recover by restarting the simulation
                connect = False
                if failure_count > 50:
                    connect=True
                # try restarting the simulation, and if that doesn't work disonnect entirely then reconnect
                self.restart_sim(connect=connect)
                self.object_handles = []
                self.vrep_names = []
                self.object_colors = []
            for object_idx in range(len(self.obj_mesh_ind)):
                # if setup for capture, no need for randomization / scrambling of the blocks.
                if self.capture_logoblock_dataset:
                    curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[object_idx])
                else:
                    curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])

                if self.is_testing and self.test_preset_cases:
                    curr_mesh_file = self.test_obj_mesh_files[object_idx]
                # TODO(ahundt) define more predictable object names for when the number of objects is beyond the number of colors
                #print(f"Currently Trying to Import: {curr_mesh_file}")
                curr_shape_name = 'shape_%02d' % object_idx
                self.vrep_names.append(curr_shape_name)
                drop_x, drop_y, object_position, object_orientation = self.generate_random_object_pose()
                if self.is_testing and self.test_preset_cases:
                    object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1], self.test_obj_positions[object_idx][2]]
                    object_orientation = [self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]

                # Loading object position and orientations from an array
                if self.test_preset_arr is not None:
                    object_position = self.test_preset_arr[object_idx][0]
                    object_orientation = self.test_preset_arr[object_idx][1]

                # Set the colors in order
                #print(f"setting color at idx {object_idx} to {self.obj_mesh_color[object_idx]}") 
                object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]
                # If there are more objects than total colors this line will break,
                # applies mod to loop back to the first color.
                object_color_name = self.color_names[object_idx % len(self.color_names)]
                # add the color of this object to the list.
                self.object_colors.append(object_color_name)
                #print('Adding object: ' + curr_mesh_file + ' as ' + curr_shape_name)
                do_break = False
                ret_ints = []
                ret_resp = 0
                while len(ret_ints) == 0:
                    do_break = False
                    #print(f"obj pos {object_position}") 
                    #print(f"obj ori {object_orientation}") 
                    #print(f"obj col {object_color}") 
                    #print(curr_mesh_file)
                    #print(curr_shape_name)

                    # TODO: ZH, We don't really need this, remove this if statement after testing
                    scale = [self.obj_scale]
                    # print(object_position + object_orientation + object_color + scale)
                    if self.textured:
                        ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShapeWTextureWScale',[0, 0, 255, 0], object_position + object_orientation + object_color + scale, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
                    else:
                        ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShapeWScale',[0, 0, 255, 0], object_position + object_orientation + object_color + scale, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)

                    if ret_resp == 8:
                        print('Failed to add ' + curr_mesh_file + ' to simulation. Auto retry ' + str(failure_count))
                        failure_count += 1
                        if failure_count % 3 == 2:
                            # If a few failures happen in a row, do a simulation reset and try again
                            do_break = True
                            break
                        elif failure_count > 100:
                            print('Failed to add new objects to simulation. Quitting. Please restart manually.')
                            exit(1)
                if do_break:
                    break
                curr_shape_handle = ret_ints[0]
                self.object_handles.append(curr_shape_handle)
                if not (self.is_testing and self.test_preset_cases):
                    time.sleep(0.5)
            # we have completed the loop adding all objects!
            add_success = len(self.object_handles) == len(self.obj_mesh_ind)
        self.prev_obj_positions = []
        self.obj_positions = []

        # now reposition objects if we are unstacking
        if self.task_type == 'unstack':
            self.reposition_objects()

    def restart_sim(self, connect=False):
        if connect:
            # Connect to simulator
            failure_count = 0
            connected = False
            while not connected:
                vrep.simxFinish(-1) # Just in case, close all opened connections
                self.sim_joint_handles = []
                self.sim_client = vrep.simxStart('127.0.0.1', self.tcp_port, True, False, 5000, 1) # Connect to V-REP on port 19997
                if self.sim_client == -1:
                    failure_count += 1
                    print('Failed to connect to simulation (V-REP remote API server) on attempt ' + str(failure_count))
                    if failure_count > 10:
                        print('Could not connect, Exiting')
                        exit(1)
                else:
                    print('Connected to simulation.')
                    connected = True
                    # self.restart_sim()

        sim_ret, self.UR5_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5',vrep.simx_opmode_blocking)
        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        sim_ret, self.UR5_position_goal_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_position_goal_target',vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        sim_ok = False
        while not sim_ok: # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(0.5)
            sim_started = vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            print('sim started 1: ' + str(sim_started))
            time.sleep(0.5)
            sim_started = vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            print('sim started 2: ' + str(sim_started))
            time.sleep(0.5)
            sim_ret, self.UR5_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
            # check sim, but we are already in the restart loop so don't recurse
            sim_ok = sim_started == vrep.simx_return_ok and self.check_sim(restart_if_not_ok=False)

    def check_obj_in_scene(self, obj_handle, workspace_limits=None, buffer_meters=0.1):
        """
        Check if object/gripper specified by CoppeliaSim handle is in scene
        Arguments:
            obj_handle: CoppeliaSim object handle
            workspace_limits: Workspace limit coordinates, defaults to self.workspace_limits
            buffer_meters: Amount of buffer to allow, defaults to 0.1
        """

        if workspace_limits is None:
            workspace_limits = self.workspace_limits

        # get object position
        sim_ret, pos = vrep.simxGetObjectPosition(self.sim_client, obj_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = pos[0] > workspace_limits[0][0] - buffer_meters and pos[0] < workspace_limits[0][1] + buffer_meters and pos[1] > workspace_limits[1][0] - buffer_meters and pos[1] < workspace_limits[1][1] + buffer_meters and pos[2] > workspace_limits[2][0] and pos[2] < workspace_limits[2][1]

        return sim_ok

    def check_sim(self, restart_if_not_ok=True):
        # buffer_meters = 0.1  # original buffer value
        buffer_meters = 0.1
        # Check if simulation is stable by checking if gripper is within workspace
        sim_ok = self.check_obj_in_scene(self.UR5_tip_handle)

        if restart_if_not_ok and not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim(connect=True)
            self.add_objects()
        return sim_ok

    def get_task_score(self):

        key_positions = np.asarray([[-0.625, 0.125, 0.0], # red
                                    [-0.625, -0.125, 0.0], # blue
                                    [-0.375, 0.125, 0.0], # green
                                    [-0.375, -0.125, 0.0]]) #yellow

        obj_positions = np.asarray(self.get_obj_positions())
        obj_positions.shape = (1, obj_positions.shape[0], obj_positions.shape[1])
        obj_positions = np.tile(obj_positions, (key_positions.shape[0], 1, 1))

        key_positions.shape = (key_positions.shape[0], 1, key_positions.shape[1])
        key_positions = np.tile(key_positions, (1 ,obj_positions.shape[1] ,1))

        key_dist = np.sqrt(np.sum(np.power(obj_positions - key_positions, 2), axis=2))
        key_nn_idx = np.argmin(key_dist, axis=0)

        return np.sum(key_nn_idx == np.asarray(range(self.num_obj)) % 4)


    def check_goal_reached(self):

        goal_reached = self.get_task_score() == self.num_obj
        return goal_reached


    def stop_sim(self):
        if self.is_sim:
            # Now send some data to V-REP in a non-blocking fashion:
            # vrep.simxAddStatusbarMessage(sim_client,'Hello V-REP!',vrep.simx_opmode_oneshot)

            # # Start the simulation
            # vrep.simxStartSimulation(sim_client,vrep.simx_opmode_oneshot_wait)

            # # Stop simulation:
            # vrep.simxStopSimulation(sim_client,vrep.simx_opmode_oneshot_wait)

            # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
            vrep.simxGetPingTime(self.sim_client)

            # Now close the connection to V-REP:
            vrep.simxFinish(self.sim_client)


    def get_obj_positions(self, relative_to_handle=-1):
        if not self.is_sim:
            raise NotImplementedError('get_obj_positions() only supported in simulation, if you are training stacking try specifying --check_z_height')
        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, relative_to_handle, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def get_objects_in_scene(self, workspace_limits=None, buffer_meters=0.0):
        """
        Function to iterate through all object positions and return number of objects within workspace_limits
        Returns:
            objs: list of CoppeliaSim object handles in scene
        """

        if workspace_limits is None:
            workspace_limits = self.workspace_limits

        # iterate through self.object_handles and check if in scene
        return [obj for obj in self.object_handles if self.check_obj_in_scene(obj, workspace_limits=workspace_limits, buffer_meters=buffer_meters)]

    def get_obj_positions_and_orientations(self):
        if not self.is_sim:
            raise NotImplementedError('get_obj_positions_and_orientations() only supported in simulation')

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations

    def action_heightmap_coordinate_to_3d_robot_pose(self, x_pixel, y_pixel, action_name, valid_depth_heightmap, robot_push_vertical_offset=0.026):
        # Adjust start position of all actions, and make sure z value is safe and not too low
        def get_local_region(heightmap, region_width=0.03):
            safe_kernel_width = int(np.round((region_width/2)/self.heightmap_resolution))
            return heightmap[max(y_pixel - safe_kernel_width, 0):min(y_pixel + safe_kernel_width + 1, heightmap.shape[0]), max(x_pixel - safe_kernel_width, 0):min(x_pixel + safe_kernel_width + 1, heightmap.shape[1])]

        finger_width = 0.04
        finger_touchdown_region = get_local_region(valid_depth_heightmap, region_width=finger_width)
        safe_z_position = self.workspace_limits[2][0]
        if finger_touchdown_region.size != 0:
            safe_z_position += np.max(finger_touchdown_region)
        else:
            safe_z_position += valid_depth_heightmap[y_pixel][x_pixel]
        if self.background_heightmap is not None:
            # add the height of the background scene
            safe_z_position += np.max(get_local_region(self.background_heightmap, region_width=0.03))
        push_may_contact_something = False

        if action_name == 'push':
            # determine if the safe z position might actually contact anything during the push action
            # TODO(ahundt) common sense push motion region can be refined based on the rotation angle and the direction of travel
            push_width = 0.2
            local_push_region = get_local_region(valid_depth_heightmap, region_width=push_width)
            # push_may_contact_something is True for something noticeably higher than the push action z height
            max_local_push_region = np.max(local_push_region)
            if max_local_push_region < 0.01:
                # if there is nothing more than 1cm tall, there is nothing to push
                push_may_contact_something = False
            else:
                push_may_contact_something = safe_z_position - self.workspace_limits[2][0] + robot_push_vertical_offset < max_local_push_region
            # print('>>>> Gripper will push at height: ' + str(safe_z_position) + ' max height of stuff: ' + str(max_local_push_region) + ' predict contact: ' + str(push_may_contact_something))
            push_str = ''
            if not push_may_contact_something:
                push_str += 'Predicting push action failure, heuristics determined '
                push_str += 'push at height ' + str(safe_z_position)
                push_str += ' would not contact anything at the max height of ' + str(max_local_push_region)
                print(push_str)

        primitive_position = [x_pixel * self.heightmap_resolution + self.workspace_limits[0][0], y_pixel * self.heightmap_resolution + self.workspace_limits[1][0], safe_z_position]
        return primitive_position, push_may_contact_something

    def reposition_objects(self, unstack_drop_height=0.05, action_log=None, logger=None,
            goal_condition=None, workspace_limits=None):
        # grasp blocks from previously placed positions and place them in a random position.
        if self.place_task and self.unstack:
            print("------- UNSTACKING --------")

            if len(self.place_pose_history) == 0:
                print("NO PLACE HISTORY TO UNSTACK YET.")
                print("HUMAN, PLEASE MOVE BLOCKS AROUND")
                print("SLEEPING FOR 10 SECONDS")

                for i in range(10):
                    print("SLEEPING FOR %d" % (10 - i))
                    time.sleep(1)

                print("-------- RESUMING AFTER MANUAL UNSTACKING --------")

            else:
                place_pose_history = self.place_pose_history.copy()
                place_pose_history.reverse()

                # unstack the block on the bottom of the stack so that the robot doesn't keep stacking in the same spot.
                place_pose_history.append(place_pose_history[-1])

                holding_object = not(self.close_gripper())
                # if already has an object in the gripper when reposition objects gets called, place that object somewhere random
                if holding_object:
                    _, _, rand_position, rand_orientation = self.generate_random_object_pose()
                    rand_position[2] = unstack_drop_height  # height from which to release blocks (0.05 m per block)
                    rand_angle = rand_orientation[0]

                    self.place(rand_position, rand_angle, save_history=False)
                else:
                    self.open_gripper()

                # go to x,y position of previous places and pick up the max_z height from the depthmap (top of the stack)
                for pose in place_pose_history:
                    x, y, z, angle = pose

                    valid_depth_heightmap, color_heightmap, depth_heightmap, max_z_height, color_img, depth_img = self.get_camera_data(return_heightmaps=True)

                    # get depth_heightmap pixel_coordinates of where the previous place was
                    x_pixel = int((x - self.workspace_limits[0][0]) / self.heightmap_resolution)
                    x_pixel = min(x_pixel, 223)  # prevent indexing outside the heightmap bounds

                    y_pixel = int((y - self.workspace_limits[1][0]) / self.heightmap_resolution)
                    y_pixel = min(y_pixel, 223)

                    primitive_position, _ = self.action_heightmap_coordinate_to_3d_robot_pose(x_pixel, y_pixel, 'grasp', valid_depth_heightmap)

                    # this z position is checked based on the x,y position of the robot. Previously, the z height was the max z_height in the depth_heightmap
                    # plus an offset. There
                    z = primitive_position[2]

                    grasp_success, color_success = self.grasp([x, y, z], angle)
                    if grasp_success:
                        _, _, rand_position, rand_orientation = self.generate_random_object_pose()
                        rand_position[2] = unstack_drop_height  # height from which to release blocks (0.05 m per block)
                        rand_angle = rand_orientation[0]

                        self.place(rand_position, rand_angle, save_history=False)

                # clear the place hisory after unstacking
                self.place_pose_history = []

            print("------- UNSTACKING COMPLETE --------")

        else:
            if self.is_sim:
                # Move gripper out of the way to the home position
                success = self.go_home()
                if not success:
                    return success
                # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
                # vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
                # time.sleep(1)

                # if not unstacking, place all objects randomly
                if self.task_type is None or self.task_type != 'unstack':
                    for object_handle in self.object_handles:
                        # Drop object at random x,y location and random orientation in robot workspace
                        self.reposition_object_randomly(object_handle)
                        time.sleep(0.5)

                # if unstacking, need to create a stack at a random location
                else:
                    successful_stack = False
                    while not successful_stack:
                        # get random obj pose
                        _, _, obj_pos, obj_ori = self.generate_random_object_pose()
                        obj_ori[:2] = [0, 0]

                        # iterate through remaining object handles and place them on top of existing stack (randomize order)
                        obj_handles_rand = np.arange(len(self.object_handles))
                        np.random.shuffle(obj_handles_rand)
                        for ind, i in enumerate(obj_handles_rand):
                            # reposition object as (x,y) position of first block, set z pos depending on stack height
                            # same orientation (TODO(adit98) add noise later?)
                            obj_pos[-1] = ind * 0.06 + 0.05

                            # reposition object
                            self.reposition_object_at_list_index_to_location(obj_pos, obj_ori, i)

                            # regenerate object orientation and keep only the top block rotation
                            _, _, _, obj_ori = self.generate_random_object_pose()
                            obj_ori[:2] = [0, 0]

                            # wait for objects to settle
                            time.sleep(0.75)

                        # continue to retry until we have a successful stack of 4 blocks
                        successful_stack, stack_height = self.check_stack(np.ones(len(self.object_handles)))
                        print('reposition_objects(): successful stack:', successful_stack, 'stack_height:', stack_height)
                        if stack_height >= len(self.object_handles):
                            successful_stack = True

                # an extra half second so things settle down
                time.sleep(0.5)

                return True

    def get_camera_data(self, workspace_limits=None, heightmap_resolution=None, return_heightmaps=False, go_home=True, z_height_retake_threshold=0.3, median_filter_size=5, color_median_filter_size=5):
        """
        # Returns

        [valid_depth_heightmap, color_heightmap, depth_heightmap, max_z_height, color_img, depth_img] if return_heightmaps is True, otherwise [color_img, depth_img]

        """
        if workspace_limits is None:
            workspace_limits = self.workspace_limits

        if heightmap_resolution is None:
            heightmap_resolution = self.heightmap_resolution

        max_z_height = np.inf
        if go_home:
            self.go_home(block_until_home=True)

        def get_color_depth():
            """ Get the raw color and depth images
            """
            if self.is_sim:
                sim_ret = None
                while sim_ret != vrep.simx_return_ok:
                    # Get color image from simulation
                    sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
                color_img = np.asarray(raw_image)
                color_img.shape = (resolution[1], resolution[0], 3)
                color_img = color_img.astype(np.float)/255
                color_img[color_img < 0] += 1
                color_img *= 255
                color_img = np.fliplr(color_img)
                color_img = color_img.astype(np.uint8)

                sim_ret = None
                while sim_ret != vrep.simx_return_ok:
                    # Get depth image from simulation
                    sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
                depth_img = np.asarray(depth_buffer)
                depth_img.shape = (resolution[1], resolution[0])
                depth_img = np.fliplr(depth_img)
                zNear = 0.01
                zFar = 10
                depth_img = depth_img * (zFar - zNear) + zNear

            else:
                # prevent camera from taking a picture while the robot is still in the frame.

                # Get color and depth image from ROS service
                color_img, depth_img = self.camera.get_data()
                depth_img = depth_img.astype(float) / 1000 # unit: mm -> meter
                # color_img = self.camera.color_data.copy()
                # depth_img = self.camera.depth_data.copy()
            return color_img, depth_img
        color_img, depth_img  = get_color_depth() # unit: mm -> meter

        if return_heightmaps:
            # this allows the error to print only once, so it doesn't spam the console.
            print_error = 0

            while max_z_height > z_height_retake_threshold:
                scaled_depth_img = depth_img * self.cam_depth_scale  # Apply depth scale from calibration
                color_heightmap, depth_heightmap = utils.get_heightmap(color_img, scaled_depth_img, self.cam_intrinsics, self.cam_pose,
                                                                    workspace_limits, heightmap_resolution, background_heightmap=self.background_heightmap,
                                                                    median_filter_pixels=median_filter_size, color_median_filter_pixels=color_median_filter_size)
                # TODO(ahundt) switch to masked array, then only have a regular heightmap
                valid_depth_heightmap = depth_heightmap.copy()
                valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

                _, max_z_height, _ = self.check_z_height(valid_depth_heightmap, reward_multiplier=1)

                # If just manipulating blocks for dataset image generation, no need to check height.
                if self.capture_logoblock_dataset:
                    break

                if max_z_height > z_height_retake_threshold:
                    if print_error > 3:
                        print('ERROR: depth_heightmap value too high. '
                              'Use the UR5 teach mode to move the robot manually to the home position. '
                              'max_z_height: ', max_z_height)

                    # Get color and depth image from ROS service
                    color_img, depth_img = get_color_depth()
                    print_error += 1
                    time.sleep(0.1)


            return valid_depth_heightmap, color_heightmap, depth_heightmap, max_z_height, color_img, depth_img

        # if not return_depthmaps, return just raw images
        return color_img, depth_img


    def parse_tcp_state_data(self, state_data, subpackage):
        """
        state_data: 'joint_data', 'cartesian_info', 'force_mode_data', 'tool_data'
        """
        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0];
        robot_message_type = data_bytes[4]
        assert(robot_message_type == 16)
        byte_idx = 5

        # Parse sub-packages
        subpackage_types = {'joint_data' : 1, 'cartesian_info' : 4, 'force_mode_data' : 7, 'tool_data' : 2}
        while byte_idx < data_length:
            # package_length = int.from_bytes(data_bytes[byte_idx:(byte_idx+4)], byteorder='big', signed=False)
            package_length = struct.unpack("!i", data_bytes[byte_idx:(byte_idx+4)])[0]
            byte_idx += 4
            package_idx = data_bytes[byte_idx]
            if package_idx == subpackage_types[subpackage]:
                byte_idx += 1
                break
            byte_idx += package_length - 4

        def parse_joint_data(data_bytes, byte_idx):
            actual_joint_positions = [0,0,0,0,0,0]
            target_joint_positions = [0,0,0,0,0,0]
            for joint_idx in range(6):
                actual_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
                target_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+8):(byte_idx+16)])[0]
                byte_idx += 41
            return actual_joint_positions

        def parse_cartesian_info(data_bytes, byte_idx):
            actual_tool_pose = [0,0,0,0,0,0]
            for pose_value_idx in range(6):
                actual_tool_pose[pose_value_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
                byte_idx += 8
            return actual_tool_pose

        def parse_tool_data(data_bytes, byte_idx):
            byte_idx += 2
            tool_analog_input2 = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            return tool_analog_input2

        parse_functions = {'joint_data' : parse_joint_data, 'cartesian_info' : parse_cartesian_info, 'tool_data' : parse_tool_data}
        return parse_functions[subpackage](data_bytes, byte_idx)

    def parse_rtc_state_data(self, state_data):

        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0]
        assert(data_length == 812)
        byte_idx = 4 + 8 + 8*48 + 24 + 120
        TCP_forces = [0,0,0,0,0,0]
        for joint_idx in range(6):
            TCP_forces[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            byte_idx += 8

        return TCP_forces


    def close_gripper(self, nonblocking=False):
        """
        # Arguments

        nonblocking: If true, the function will not wait for the robot to finish its action, it will return immediately.
                     If false, the function will wait for the robot to finish its action, then return the result.

        # Return

        True if the gripper is fully closed at the end of the call, false otherwise.
        The gripper can take a full second to close, but this function may return in 1 ms
        May return False if nonblocking is True because an open gripper will not have closed completely yet,
        even if the gripper eventually closes all the way.
        """
        if self.is_sim:
            gripper_motor_velocity = -0.5
            gripper_motor_force = 100
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
            gripper_fully_closed = False
            while gripper_joint_position > -0.045: # Block until gripper is fully closed, value previously -0.047
                sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
                print('gripper position: ' + str(gripper_joint_position))
                if new_gripper_joint_position >= gripper_joint_position:
                    return gripper_fully_closed
                gripper_joint_position = new_gripper_joint_position
            gripper_fully_closed = True

        elif self.gripper is None:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "set_digital_out(8,True)\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()
            if nonblocking:
                gripper_fully_closed = True
            else:
                time.sleep(1.5)
                gripper_fully_closed =  self.check_grasp()
        else:
            # stop is done to clear the current state,
            # for cases like running close twice in a row
            # to first actually grasp an object then
            # to second check if the object is still present
            self.gripper.stop(block=not nonblocking, timeout=0.1)
            # actually close the gripper
            self.gripper.close(block=not nonblocking)
            if nonblocking:
                gripper_fully_closed = True
            else:
                gripper_fully_closed = not self.gripper.object_detected()

        return gripper_fully_closed


    def open_gripper(self, nonblocking=False, timeout_seconds=5):
        """
        # Returns

        True if the gripper is open after the call, otherwise false.
        """
        if self.is_sim:
            gripper_motor_velocity = 0.5
            gripper_motor_force = 20
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
            time_start = time.time()
            while gripper_joint_position < 0.03: # Block until gripper is fully open
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
                time_snapshot = time.time()
                if time_snapshot - time_start > timeout_seconds:
                    return False
            return True

        elif self.gripper is None:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "set_digital_out(8,False)\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()
            if not nonblocking:
                time.sleep(1.5)
        else:
            self.gripper.open(block=not nonblocking)
            return self.gripper.is_opened()


    def get_state(self):

        state_data = None
        while state_data is None:
            try:
                self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_socket.settimeout(1.0)
                self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
                state_data = self.tcp_socket.recv(2048)
            except socket.timeout as e:
                print('WARNING: robot.py get_state() TIMEOUT ' + str(e))
            except TimeoutError as e:
                print('WARNING: robot.py get_state() TIMEOUT' + str(e))
                pass
            self.tcp_socket.close()
        return state_data


    def move_to(self, tool_position, tool_orientation=None, timeout_seconds=10, heightmap_rotation_angle=None, legacy_mode=True, sim_move_step=0.01):
        """
        legacy_mode: bool, Legacy mode manually increments the gripper position, rather than using simulator motion commands.
        Note to use legacy mode in the simulator you need to go into the simulation and disable the "threaded child script"
        associated with the object UR5_position_goal_target.
        sim_move_step: How far the simulated robot should mobe per time step, very large is 0.05, small is 0.01, we use 0.02 at the time of writing
        """
        if np.isnan(tool_position).any():
            print('ERROR: robot.move_to() NaN encountered in goal tool_position, skipping action. Traceback of code location:')
            traceback.print_stack()
            return False

        if self.is_sim:

            # note there are 3 approaches to moving in sim below.
            # orientation only mode
            motion_mode = 3
            tool_rotation_angle = None
            if tool_orientation is None and heightmap_rotation_angle is not None:
                # Compute tool orientation from heightmap rotation angle
                tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2
                if not legacy_mode:
                    # set the goal orientation if we are using a newer moving mode,
                    # otherwise leave it None because the original way to move in sim
                    # only accepts heightmap_rotation_angle
                    tool_orientation = [np.pi/2, tool_rotation_angle, np.pi/2]

            if not legacy_mode:

                # simulator_moves_to_goal switches between using a version that automatically detects changes in vrep,
                # and another version which tries to directly send the motion command to vrep
                simulator_moves_to_goal = False

                if simulator_moves_to_goal:
                    if tool_orientation is None and heightmap_rotation_angle is None:
                        # position only mode
                        motion_mode = 1
                        # we still need a tool orientation as dummy parameters
                        tool_orientation = [np.pi/2, 0.0, np.pi/2]
                    # here we make a call that calls moveToObject() in v-rep
                    failure_count = 0
                    ret_ints = []
                    while len(ret_ints) == 0:
                        # do_break = False
                        # see simMoveToObject for details, our script function is in the actual V-REP sim. http://coppeliarobotics.com/helpFiles/en/apiFunctions.htm#simMoveToObject
                        move_velocity = 3.0
                        move_accel = 5.0
                        # The integer configuration for the motion mode, the 4th integer in the list, is 1: position only, 2: rotation only, 3: position and rotation
                        ret_resp,ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(
                                self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'moveObjectToPose',
                                 # int params object to move, object to teleport for moving towards, base frame object, motion mode
                                [self.UR5_target_handle, self.UR5_position_goal_target_handle, self.UR5_handle, motion_mode],
                                tool_position + tool_orientation + [move_velocity, move_accel], [], bytearray(), vrep.simx_opmode_blocking)
                        if ret_resp == 8:
                            print('Failed to move gripper. Auto retry ' + str(failure_count))
                            failure_count += 1
                            # if failure_count % 3 == 2:
                            #     # If a few failures happen in a row, do a simulation reset and try again
                            #     do_break = True
                            #     break
                            if failure_count > 10:
                                print('Failed to move gripper to target.')
                                return False
                                # exit(1)
                else:
                    # here we make a call that calls moveToObject() in v-rep
                    # This first while loop is to set the target position and orientation in a single sim time step
                    failure_count = 0
                    ret_ints = []
                    while len(ret_ints) == 0:
                        # do_break = False
                        # see simMoveToObject for details, our script function is in the actual V-REP sim. http://coppeliarobotics.com/helpFiles/en/apiFunctions.htm#simMoveToObject
                        move_velocity = 3.0
                        move_accel = 5.0
                        # The integer configuration for the motion mode, the 4th integer in the list, is 1: position only, 2: rotation only, 3: position and rotation
                        ret_resp,ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(
                                self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'setObjectPose',
                                 # int params object to move, object to teleport for moving towards, base frame object, motion mode
                                [self.UR5_position_goal_target_handle, -1, motion_mode],
                                [tool_position[0], tool_position[1], tool_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2]],
                                [], bytearray(), vrep.simx_opmode_blocking)
                        if ret_resp == 8:
                            print('Failed to move gripper. Auto retry ' + str(failure_count))
                            failure_count += 1
                            # if failure_count % 3 == 2:
                            #     # If a few failures happen in a row, do a simulation reset and try again
                            #     do_break = True
                            #     break
                            if failure_count > 10:
                                print('Failed to move gripper to target.')
                                return False
                                # exit(1)
                    # time.sleep(0.1) # give simulator a moment to settle and prevent teleportation
                    # # Set the position then the orientation one after the other
                    # if motion_mode == 1 or motion_mode == 3:
                    #     vrep.simxSetObjectPosition(self.sim_client, self.UR5_position_goal_target_handle, -1, tool_position,vrep.simx_opmode_blocking)
                    # if motion_mode == 2 or motion_mode == 3:
                    #     vrep.simxSetObjectOrientation(self.sim_client, self.UR5_position_goal_target_handle, -1, tool_orientation, vrep.simx_opmode_blocking)
                    # Wait until the gripper is close to the goal
                    time_start = time.time()
                    sim_ret, gripper_goal_dist = vrep.simxGetObjectPosition(self.sim_client, self.UR5_tip_handle, self.UR5_position_goal_target_handle, vrep.simx_opmode_blocking)
                    sim_ret, gripper_goal_rot = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_tip_handle, self.UR5_position_goal_target_handle, vrep.simx_opmode_blocking)
                    gripper_combined = gripper_goal_dist + gripper_goal_rot
                    while not np.all(np.abs(gripper_combined) < np.array(self.tool_pose_tolerance)):
                        sim_ret, gripper_goal_dist = vrep.simxGetObjectPosition(self.sim_client, self.UR5_tip_handle, self.UR5_position_goal_target_handle, vrep.simx_opmode_blocking)
                        sim_ret, gripper_goal_rot = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_tip_handle, self.UR5_position_goal_target_handle, vrep.simx_opmode_blocking)
                        gripper_combined = gripper_goal_dist + gripper_goal_rot
                        time_snapshot = time.time()
                        if time_snapshot - time_start > timeout_seconds:
                            print('robot.move_to() timeout, robot did not reach goal')
                            return False
                        time.sleep(0.01) # give simulator a moment to settle and prevent teleportation
                    return True
            else:
                if tool_orientation is not None:
                    raise NotImplementedError('move_to() tool_orientation is not supported when is stepping to arbitrary orientations')
                # this is the original way sim motion was handled, the absolute position of the motion target dummy
                # is incremented linearly with each time step. However this is very slow and can result in stability
                # problems when there is a collision.
                # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
                if np.isnan(UR5_target_position).any():
                    return False

                move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
                move_magnitude = np.linalg.norm(move_direction)
                # prevent division by 0, source: https://stackoverflow.com/a/37977222/99379
                move_step = sim_move_step * np.divide(move_direction, move_magnitude, out=np.zeros_like(move_direction), where=move_magnitude!=0)

                num_move_steps = np.divide(move_direction, move_step, out=np.zeros_like(move_direction), where=move_step!=0)
                num_move_steps = int(np.max(np.floor(num_move_steps)))

                num_rotation_steps = 1
                if tool_rotation_angle is not None:
                    # Compute gripper orientation and rotation increments
                    sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
                    rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
                    num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

                # Simultaneously move and rotate gripper if an orientation is provided, only translate otherwise
                for step_iter in range(max(num_move_steps, num_rotation_steps)):
                    if step_iter <= num_move_steps:
                        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                    if tool_rotation_angle is not None and step_iter <= num_rotation_steps:
                        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter, num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
                if tool_rotation_angle is not None:
                    vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

                return True
        else:
            # move the real robot to a position.
            if tool_orientation is None and heightmap_rotation_angle is None:
                # If no orientation is provided, use the current one.
                actual_pose = self.get_cartesian_position()
                tool_orientation = actual_pose[3:]
            elif tool_orientation is None and heightmap_rotation_angle is not None:
                # Compute tool orientation from heightmap rotation angle
                grasp_orientation = [1.0,0.0]
                if heightmap_rotation_angle > np.pi:
                    heightmap_rotation_angle = heightmap_rotation_angle - 2*np.pi
                tool_rotation_angle = heightmap_rotation_angle/2
                tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
            print('pose before: ' + str(tool_position) + str(tool_orientation))
            tool_position_tcp, tool_orientation_tcp = gripper_control_pose_to_arm_control_pose(tool_position, tool_orientation, self.tool_tip_to_gripper_center_transform)
            print('pose after: ' + str(tool_position_tcp) + str(tool_orientation_tcp))
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            # tcp_command = "set_tcp(p[-0.1,0.0,0.0,0.0,0.0,0.0])\n"
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (tool_position_tcp[0],tool_position_tcp[1],tool_position_tcp[2],tool_orientation_tcp[0],tool_orientation_tcp[1],tool_orientation_tcp[2],self.tool_acc,self.tool_vel)
            print(tcp_command)
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()

            # Block until robot reaches target tool position
            return self.block_until_cartesian_position(tool_position_tcp, timeout_seconds=timeout_seconds)

    def guarded_move_to(self, tool_position, tool_orientation):
        if self.is_sim:
            raise NotImplementedError

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))

        # Read actual tool position
        tcp_state_data = self.tcp_socket.recv(2048)
        actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
        execute_success = True

        # Increment every cm, check force
        self.tool_acc = 0.1 # 1.2 # 0.5

        while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]

            # Compute motion trajectory in 1cm increments
            increment = np.asarray([(tool_position[j] - actual_tool_pose[j]) for j in range(3)])
            if np.linalg.norm(increment) < 0.01:
                increment_position = tool_position
            else:
                increment = 0.01*increment/np.linalg.norm(increment)
                increment_position = np.asarray(actual_tool_pose[0:3]) + increment

            # TODO(ahundt) apply tcp transform to guarded_move()
            # Move to next increment position (blocking call)
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (increment_position[0],increment_position[1],increment_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.tool_acc,self.tool_vel)
            self.tcp_socket.send(str.encode(tcp_command))

            time_start = time.time()
            tcp_state_data = self.tcp_socket.recv(2048)
            actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            while not all([np.abs(actual_tool_pose[j] - increment_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                # print([np.abs(actual_tool_pose[j] - increment_position[j]) for j in range(3)])
                tcp_state_data = self.tcp_socket.recv(2048)
                actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
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
                print('Warning: contact detected! Movement halted. TCP forces: [%f, %f, %f]' % (TCP_forces[0], TCP_forces[1], TCP_forces[2]))
                execute_success = False
                break

            time.sleep(0.01)

        self.tool_acc = 1.2 # 1.2 # 0.5

        self.tcp_socket.close()
        self.rtc_socket.close()

        return execute_success


    def move_joints(self, joint_configuration, timeout_seconds=7):
        if self.is_sim:
            if not self.sim_joint_handles:
                # set all of the joint handles
                for i in range(1,6):
                    sim_ret, joint_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_joint' + str(i), vrep.simx_opmode_blocking)
                    self.sim_joint_handles += [joint_handle]
            for handle, position in zip(self.sim_joint_handles, joint_configuration):
                sim_ret= vrep.simxSetJointPosition(self.sim_client, handle, position, vrep.simx_opmode_blocking)
            return True
        else:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "movej([%f" % joint_configuration[0]
            for joint_idx in range(1,6):
                tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
            tcp_command = tcp_command + "],a=%f,v=%f)\n" % (self.joint_acc, self.joint_vel)
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()

            # Block until robot reaches home state
            state_data = self.get_state()
            actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
            time_start = time.time()
            while not all([np.abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
                state_data = self.get_state()
                actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
                time.sleep(0.01)
                time_snapshot = time.time()
                if time_snapshot - time_start > timeout_seconds:
                    print('move_joints() Timeout')
                    return False
            return True


    def go_home(self, block_until_home=False, timeout_seconds=7):
        if self.is_sim:
            success = self.move_to(self.sim_home_position, None)
            if not self.home_joint_config:
                return success
            else:
                # hard set the joint position to the home position to work around IK choosing
                # elbow down positions, which leads to physically impossible simulator states.
                print(f"moving to home joint config {self.home_joint_config}") 
                return self.move_joints(self.home_joint_config)
        else:
            self.move_joints(self.home_joint_config)
            if not block_until_home:
                timeout_seconds = 0
            # block_until_home with 0 second timeout will just
            # get data from the robot once to indicate if we are
            # without blocking otherwise.
            return self.block_until_home(timeout_seconds)


    def check_grasp(self):
        """
        Note: must be preceded by close_gripper()
        """
        if self.is_sim:
            raise NotImplementedError

        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        return tool_analog_input2 > 0.26

    # HK: added a function to check if the right color is grasped
    def check_correct_color_grasped(self, color_ind):
        '''
        color_ind: the index in color_names to grasp.
        '''
        object_positions = np.asarray(self.get_obj_positions())
        object_positions = object_positions[:,2]
        grasped_object_ind = np.argmax(object_positions)
        grasped_object_handle = self.object_handles[grasped_object_ind]
        # color_index = np.where(color==1)
        # if grasped_object_ind == color_index[0]:
        if grasped_object_ind == color_ind:
            return True
        else:
            return False


    def get_highest_object_list_index_and_handle(self):
        """
        Of the objects in self.object_handles, get the one with the highest z position and its handle.

        # Returns

           grasped_object_ind, grasped_object_handle
        """
        object_positions = np.asarray(self.get_obj_positions())
        object_positions = object_positions[:,2]
        grasped_object_ind = np.argmax(object_positions)
        grasped_object_handle = self.object_handles[grasped_object_ind]
        return grasped_object_ind, grasped_object_handle


    def reposition_objects_near_gripper(self, distance_threshold=0.1, put_inside_workspace=True):
        """ Simulation only function to detect objects near the gripper.

            put_inside_workspace: True will select a random position inside workspace, false will select a specific position outside the workspace

            # Returns

            True if there are objects in the scene, False otherwise.
        """
        object_positions = np.asarray(self.get_obj_positions(self.UR5_tip_handle))
        if object_positions.shape[0] == 0:
            return False
        else:
            is_near_gripper = np.linalg.norm(object_positions, axis=1) < distance_threshold
            # get_object_handles_near_gripper
            if put_inside_workspace:
                object_handles_near_gripper = np.array(self.object_handles)[is_near_gripper]
                for handle in object_handles_near_gripper:
                    self.reposition_object_randomly(handle)
            else:
                for i, is_near in enumerate(is_near_gripper):
                    # set all objects near the gripper to a coordinated position
                    if is_near:
                        vrep.simxSetObjectPosition(self.sim_client,self.object_handles[i],-1,(-0.5, 0.5 + 0.05*float(i), 0.1),vrep.simx_opmode_blocking)
            return True


    # Primitives ----------------------------------------------------------

    def grasp(self, position, heightmap_rotation_angle, object_color=None, workspace_limits=None, go_home=True):
        """
        object_color: The index in the list self.color_names expected for the object to be grasped. If object_color is None the color does not matter.
        """
        if workspace_limits is None:
            workspace_limits = self.workspace_limits
        print('Executing: grasp at (%f, %f, %f) orientation: %f' % (position[0], position[1], position[2], heightmap_rotation_angle))
        position = np.asarray(position).copy()

        if self.is_sim:

            # Avoid collision with floor
            position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)

            # Move gripper to location above grasp target, this is the pre-grasp and post-grasp height
            # grasp_location_margin = 0.15
            grasp_location_margin = 0.2
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_grasp_target
            self.move_to(tool_position, heightmap_rotation_angle=heightmap_rotation_angle)

            # Ensure gripper is open
            self.open_gripper()

            # Approach grasp target
            self.move_to(position, None)

            # Close gripper to grasp target
            gripper_fully_closed = self.close_gripper()

            # Move gripper to location above grasp target
            self.move_to(location_above_grasp_target, None)
            # move to the simulator home position
            if go_home:
                self.go_home()

            # Check if grasp is successful
            gripper_fully_closed = self.close_gripper()
            grasp_success = not gripper_fully_closed

            # HK: Check if right color is grasped
            color_success = False
            if grasp_success and self.grasp_color_task:
                color_success = self.check_correct_color_grasped(object_color)
                print('Correct color was grasped: ' + str(color_success))

            # HK: Place grasped object at a random place
            if grasp_success:
                if self.place_task:
                    return grasp_success, color_success
                else:
                    # we are pushing and grasping, so move the objects outside the workspace
                    objects_anywhere_in_scene = self.reposition_objects_near_gripper(put_inside_workspace=False)
                    if not objects_anywhere_in_scene:
                        # there are no objects in the scene, so the grasp could not be successful
                        return False, False
        else:
            # Warning: "Real Good Robot!" specific hack, increase gripper height for our different mounting config
            # position[2] += self.gripper_ee_offset - 0.01
            position[2] -= 0.04
            # Compute tool orientation from heightmap rotation angle
            grasp_orientation = [1.0,0.0]
            if heightmap_rotation_angle > np.pi:
                heightmap_rotation_angle = heightmap_rotation_angle - 2*np.pi
            tool_rotation_angle = heightmap_rotation_angle/2
            tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
            # tool_orientation_angle = np.linalg.norm(tool_orientation)
            # tool_orientation_axis = tool_orientation/tool_orientation_angle
            # tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]
            tilted_tool_orientation = tool_orientation

            # Attempt grasp
            # find position halfway between the current and final.
            print("Grasp position before applying workspace bounds: " + str(position))
            position[2] = max(position[2], workspace_limits[2][0] + 0.04)
            position[2] = min(position[2], workspace_limits[2][1] - 0.01)
            up_pos = np.array([position[0],position[1],position[2]+0.1])

            position_tcp, tool_orientation_tcp = gripper_control_pose_to_arm_control_pose(position, tool_orientation, self.tool_tip_to_gripper_center_transform)
            # we assume the same orientation will be fine for both the position and up position
            up_pos_tcp, _ = gripper_control_pose_to_arm_control_pose(up_pos, tool_orientation, self.tool_tip_to_gripper_center_transform)

            print("Real Good Robot grasping at: " + str(position) + ", " + str(tool_orientation))
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "def process():\n"
            tcp_command += " set_digital_out(8,False)\n"
            if go_home:
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (self.home_cart_low[0],self.home_cart_low[1],self.home_cart_low[2],tool_orientation_tcp[0],tool_orientation_tcp[1],0.0,self.joint_acc,self.joint_vel)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (up_pos_tcp[0],up_pos_tcp[1],up_pos_tcp[2],tool_orientation_tcp[0],tool_orientation_tcp[1],0.0,self.joint_acc,self.joint_vel)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (position_tcp[0],position_tcp[1],position_tcp[2],tool_orientation_tcp[0],tool_orientation_tcp[1],0.0,self.joint_acc*0.1,self.joint_vel*0.1)
            tcp_command += " set_digital_out(8,True)\n"
            tcp_command += "end\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()

            self.block_until_cartesian_position(position_tcp)
            time.sleep(0.1)
            gripper_fully_closed = self.close_gripper()
            color_success = None
            bin_position = np.array([0.33, 0.40, 0.33])

            # If gripper is open, drop object in bin and check if grasp is successful
            grasp_success = False
            if not gripper_fully_closed:
                print("Possible Grasp success, moving up then closing again to see if object is still present...")
                # self.move_to([position[0],position[1],bin_position[2] - 0.14],[tool_orientation[0],tool_orientation[1],0.0])
                self.move_to(up_pos,[tool_orientation[0],tool_orientation[1],0.0])
                grasp_success = not self.close_gripper()
                if grasp_success and not self.place_task:
                    print("Grasp success, moving to drop object in bin...")
                    # Move towards the bin, and up to the drop height
                    move_waypoint = (bin_position - position) * 0.6 + position
                    move_waypoint[2] = bin_position[2]
                    self.move_to(move_waypoint, [tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2]])
                    # Move over the bin
                    self.move_to([bin_position[0],bin_position[1],bin_position[2]], [tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2]])

            else:
                print("Grasp failure, moving to home position...")

            if not grasp_success or not self.place_task:
                self.open_gripper(nonblocking=True)

            if self.place_task:
                # go back to the grasp up pos
                self.move_to(up_pos,[tool_orientation[0],tool_orientation[1],0.0])
                time.sleep(0.1)
            if go_home:
                self.go_home(block_until_home=True)

        # TODO: change to 1 and 2 arguments
        return grasp_success, color_success

    def get_midpoint(self, pos):
        """ Gets the cartesian midpoint between the current real robot position and some goal position
        """
        if self.is_sim:
            raise NotImplementedError
        # Get current tool pose
        state_data = self.get_state()
        actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
        # midpos is a hack to prevent the robot from going to an elbow down position on a long move,
        # which leads to a security stop and an end to the program run.
        midpos = (np.array(actual_tool_pose[:3]) + pos) / 2.0
        return midpos

    def push(self, position, heightmap_rotation_angle, workspace_limits=None, go_home=True):
        if workspace_limits is None:
            workspace_limits = self.workspace_limits
        print('Executing: Push at (%f, %f, %f) angle: %f' % (position[0], position[1], position[2], heightmap_rotation_angle))
        # Warning: "Real Good Robot!" specific hack, increase gripper height for our different mounting config
        position = np.asarray(position).copy()
        position[2] += self.push_vertical_offset

        # Compute push direction and endpoint (push to right of rotated heightmap)
        position, up_pos, push_endpoint, push_direction, tool_orientation, tilted_tool_orientation = push_poses(heightmap_rotation_angle, position, workspace_limits,
                                                                                                                gripper_to_arm_transform=self.tool_tip_to_gripper_center_transform)

        if self.is_sim:
            self.move_to(up_pos, heightmap_rotation_angle=heightmap_rotation_angle)

            # Ensure gripper is closed
            self.close_gripper()

            # Approach pushing point
            self.move_to(position, None)

            # Move in pushing direction towards target location, but move a bit more slowly so stuff doesn't go flying
            self.move_to(push_endpoint, None, sim_move_step=0.01)

            # Move gripper to location above grasp target
            push_success = self.move_to(up_pos, None)

            # move to the simulator home position
            if go_home:
                push_success = self.go_home()

            # Work around a simulator bug where objects will be stuck to the gripper
            objects_anywhere_in_scene = self.reposition_objects_near_gripper(put_inside_workspace=True)
            if not objects_anywhere_in_scene:
                # there are no objects in the scene, so the push could not be successful
                return False

            return push_success

        else:

            # Attempt push
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "def process():\n"
            tcp_command += " set_digital_out(8,True)\n"
            if go_home:
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (self.home_cart_low[0],self.home_cart_low[1],self.home_cart_low[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc,self.joint_vel)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (up_pos[0],up_pos[1],up_pos[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.75,self.joint_vel*0.75)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (position[0],position[1],position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (push_endpoint[0],push_endpoint[1],push_endpoint[2],tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.03)\n" % (up_pos[0],up_pos[1],up_pos[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
            if go_home:
                tcp_command += " movej([%f" % self.home_joint_config[0]
                for joint_idx in range(1,6):
                    tcp_command = tcp_command + (",%f" % self.home_joint_config[joint_idx])
                tcp_command = tcp_command + "],a=%f,v=%f,t=0,r=0.09)\n" % (self.joint_acc, self.joint_vel)
                # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
            else:
                # go to up pos instead of home
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (up_pos[0],up_pos[1],up_pos[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.75,self.joint_vel*0.75)
            tcp_command += "end\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()
            self.close_gripper(nonblocking=True)

            # Block until robot reaches target tool position and gripper fingers have stopped moving
            # state_data = self.get_state()
            # while True:
            #     state_data = self.get_state()
            #     actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            #     if all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            #         break

            # Block until robot reaches target home joint position and gripper fingers have stopped moving
            time.sleep(0.1)
            if go_home:
                push_success = self.block_until_home()
                # Redundant go home is applied in case the first move operation fails.
                if not push_success:
                    push_success = self.go_home(block_until_home=True)
            self.open_gripper(nonblocking=True)
            # time.sleep(0.25)

        return push_success

    def get_cartesian_position(self):
        if self.is_sim:
            raise NotImplementedError

        state_data = self.get_state()
        actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
        return actual_tool_pose

    def get_joint_position(self):
        """ get the position of all the joints.

        Also see the sister function move_joints()
        """
        if not self.is_sim:
            raise NotImplementedError
        joint_handles = []
        for i in range(1,6):
            sim_ret, joint_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_joint' + str(i), vrep.simx_opmode_blocking)
            joint_handles += [joint_handle]
        joint_positions = []
        for joint_handle in joint_handles:
            sim_ret, position = vrep.simxGetJointPosition(self.sim_client, joint_handle, vrep.simx_opmode_blocking)
            joint_positions += [position]
        return joint_positions

    def block_until_home(self, timeout_seconds=7):

        if self.is_sim:
            raise NotImplementedError
        # Don't wait for more than timeout_seconds,
        # but get the state from the real robot at least once.
        iteration_time_0 = time.time()
        while True:
            time_elapsed = time.time()-iteration_time_0
            state_data = self.get_state()
            actual_joint_pose = self.parse_tcp_state_data(state_data, 'joint_data')
            if all([np.abs(actual_joint_pose[j] - self.home_joint_config[j]) < self.joint_tolerance for j in range(5)]):
                print('Move to Home Position Complete')
                return True
            if int(time_elapsed) > timeout_seconds:
                print('Move to Home Position Failed')
                return False
            time.sleep(0.1)

    def block_until_cartesian_position(self, position, timeout_seconds=7):
        """Block the real program until it reaches a specified cartesian pose or the timeout in seconds.
        """
        if self.is_sim:
            raise NotImplementedError
        # Block until robot reaches target tool position and gripper fingers have stopped moving
        state_data = self.get_state()
        timeout_t0 = time.time()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            timeout_t1 = time.time()
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if ((tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and
                 all([np.abs(actual_tool_pose[j] - position[j]) < self.tool_pose_tolerance[j] for j in range(3)]))):
                return True
            if (timeout_t1 - timeout_t0) > timeout_seconds:
                return False
            tool_analog_input2 = new_tool_analog_input2

    def block_until_joint_position(self, position, timeout_seconds=7):

        if self.is_sim:
            raise NotImplementedError
        # Don't wait for more than 20 seconds
        iteration_time_0 = time.time()
        while True:
            time_elapsed = time.time()-iteration_time_0
            if int(time_elapsed) > timeout_seconds:
                print('Move to Joint Position Failed')
                return False
            state_data = self.get_state()
            actual_joint_pose = self.parse_tcp_state_data(state_data, 'joint_data')
            if all([np.abs(actual_joint_pose[j] - self.home_joint_config[j]) < self.joint_tolerance for j in range(5)]):
                print('Move to Joint Position Complete')
                return True
            time.sleep(0.1)

    def place(self, position, heightmap_rotation_angle, workspace_limits=None, distance_threshold=0.06, go_home=True, save_history=True, over_block=True):
        """ Place an object, currently only tested for blocks.

        When in sim mode it assumes the current position of the robot and grasped object is higher than any other object.
        """
        if workspace_limits is None:
            workspace_limits = self.workspace_limits

        place_pose = (position[0], position[1], position[2], heightmap_rotation_angle)
        print('Executing: Place at (%f, %f, %f) angle: %f' % place_pose)

        if save_history:
            self.place_pose_history.append(place_pose)
            while len(self.place_pose_history) > self.place_pose_history_limit:  # only store x most recent place attempts
                self.place_pose_history.pop(0)

        if self.is_sim:
            # Ensure gripper is closed
            gripper_fully_closed = self.close_gripper()
            if gripper_fully_closed:
                # There is no object present, so we cannot possibly place!
                return False
            # If the object has been grasped it should be the highest object and held by the gripper
            grasped_object_ind, grasped_object_handle = self.get_highest_object_list_index_and_handle()
            sim_ret, grasped_object_position = vrep.simxGetObjectPosition(self.sim_client, grasped_object_handle, -1, vrep.simx_opmode_blocking)
            grasped_object_position = np.array(grasped_object_position)

            # Compute tool orientation from heightmap rotation angle
            # tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2
            # tool_orientation = (np.pi/2, tool_rotation_angle, np.pi/2)
            if over_block:
                position[2] += 0.04

            # Avoid collision with floor
            position[2] = max(position[2] + 0.02, workspace_limits[2][0] + 0.02)

            # Move gripper to location above place target
            place_location_margin = 0.2
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_place_target = (position[0], position[1], position[2] + place_location_margin)
            # self.move_to(location_above_place_target, None)

            # sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
            # if tool_rotation_angle - gripper_orientation[1] > 0:
            #     increment = 0.2
            # else:
            #     increment = -0.2
            # while abs(tool_rotation_angle - gripper_orientation[1]) >= 0.2:
            #     vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + increment, np.pi/2), vrep.simx_opmode_blocking)
            #     time.sleep(0.005)
            #     sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
            # vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

            # # not supported in some sim move_to() modes: self.move_to(location_above_place_target, tool_orientation)
            self.move_to(location_above_place_target, heightmap_rotation_angle=heightmap_rotation_angle)

            # Approach place target
            self.move_to(position, None)

            # Ensure gripper is open
            self.open_gripper()

            # Move gripper to location above place target
            self.move_to(location_above_place_target, None)

            # move to the simulator home position
            if go_home:
                self.go_home()

            sim_ret, placed_object_position = vrep.simxGetObjectPosition(self.sim_client, grasped_object_handle, -1, vrep.simx_opmode_blocking)
            placed_object_position = np.array(placed_object_position)

            has_moved = np.linalg.norm(placed_object_position - grasped_object_position, axis=0) > (distance_threshold/2)
            if not has_moved:
                # The highest object, which we had supposedly grasped, didn't move!
                return False
            print('current_position: ' + str(placed_object_position))
            current_obj_z_location = placed_object_position[2]
            print('current_obj_z_location: ' + str(current_obj_z_location+distance_threshold/2))
            near_goal = np.linalg.norm(placed_object_position - position, axis=0) < (distance_threshold)
            print('goal_position: ' + str(position[2]) + ' goal_position_margin: ' + str(position[2] + place_location_margin))
            place_success = has_moved and near_goal
            print('has_moved: ' + str(has_moved) + ' near_goal: ' + str(near_goal) + ' place_success: ' + str(place_success))

            # Work around a simulator bug where objects will be stuck to the gripper
            objects_anywhere_in_scene = self.reposition_objects_near_gripper(put_inside_workspace=True)
            if not objects_anywhere_in_scene:
                # there are no objects in the scene, so the place could not be successful
                return False
            return place_success
            #if abs(current_obj_z_location - position[2]) < 0.009:
            # if ((current_obj_z_location+distance_threshold/2) >= position[2]) and ((current_obj_z_location+distance_threshold/2) < (position[2]+distance_threshold)):
            #     place_success = True
            # else:
            #     place_success = False
            # return place_success
        else:

            # Warning: "Real Good Robot!" specific hack, increase gripper height for our different mounting config
            # position[2] += self.gripper_ee_offset + 0.05
            if over_block:
                position[2] += 0.04
            # Compute tool orientation from heightmap rotation angle
            grasp_orientation = [1.0,0.0]
            if heightmap_rotation_angle > np.pi:
                heightmap_rotation_angle = heightmap_rotation_angle - 2*np.pi
            tool_rotation_angle = heightmap_rotation_angle/2
            tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
            # tool_orientation_angle = np.linalg.norm(tool_orientation)
            # tool_orientation_axis = tool_orientation/tool_orientation_angle
            # tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]
            # Attempt place
            position = np.asarray(position).copy()

            # find position halfway between the current and final.
            print("Place position before applying workspace bounds: " + str(position))
            # position[2] = max(position[2], workspace_limits[2][0] + self.gripper_ee_offset + 0.04)
            # position[2] = min(position[2], workspace_limits[2][1] + self.gripper_ee_offset - 0.01)
            position[2] = max(position[2], workspace_limits[2][0] + 0.04)
            position[2] = min(position[2], workspace_limits[2][1] - 0.01)
            up_pos = np.array([position[0],position[1],position[2]+0.1])

            position_tcp, tool_orientation_tcp = gripper_control_pose_to_arm_control_pose(position, tool_orientation, self.tool_tip_to_gripper_center_transform)
            # we assume the same orientation will be fine for both the position and up position
            up_pos_tcp, _ = gripper_control_pose_to_arm_control_pose(up_pos, tool_orientation, self.tool_tip_to_gripper_center_transform)
            print("Real Good Robot placing at: " + str(position) + ", " + str(tool_orientation))
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "def process():\n"
            tcp_command += " set_digital_out(8,False)\n"
            if go_home:
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (self.home_cart_low[0],self.home_cart_low[1],self.home_cart_low[2],tool_orientation_tcp[0],tool_orientation_tcp[1],0.0,self.joint_acc,self.joint_vel)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (up_pos_tcp[0],up_pos_tcp[1],up_pos_tcp[2],tool_orientation_tcp[0],tool_orientation_tcp[1],0.0,self.joint_acc,self.joint_vel)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (position_tcp[0],position_tcp[1],position_tcp[2],tool_orientation_tcp[0],tool_orientation_tcp[1],0.0,self.joint_acc*0.1,self.joint_vel*0.1)
            tcp_command += " set_digital_out(8,True)\n"
            tcp_command += "end\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()
            time.sleep(0.1)
            self.block_until_cartesian_position(position_tcp)
            time.sleep(0.1)

            self.open_gripper(nonblocking=True)
            move_to_result = self.move_to(up_pos)
            # TODO(ahundt) save previous and new depth image, and if the depth at the place coordinate increased, return True for place success
            if go_home:
                # TODO(ahundt) confirm redundant go_home works around some cases where the robot fails to reach the destination
                home_success = self.go_home(block_until_home=True)
                if not home_success:
                    home_success = self.go_home(block_until_home=True)
                return home_success
            else:
                return move_to_result

    def check_row(self, object_color_sequence,
                  num_obj=4,
                  distance_threshold=0.02,
                  separation_threshold=0.1,
                  num_directions=64,
                  check_z_height=False,
                  valid_depth_heightmap=None,
                  prev_z_height=None):
        """Check for a complete row in the correct order, along any of the `num_directions` directions.

        Input: vector length of 1, 2, or 3
        Example: goal = [0] or [0,1] or [0,1,3]

        # Arguments

        object_color_sequence: vector indicating the index order of self.object_handles we expect to grasp.
        num_obj: number of blocks in the workspace (needed to get all subsets)
        separation_threshold: The max distance cutoff between blocks in meters for the stack to be considered complete.
        distance_threshold: maximum distance for blocks to be off-row
        num_directions: number of rotations that are checked for rows.


        # Returns

        List [success, height_count].
        success: will be True if the stack matches the specified order from bottom to top, False otherwise.
        row_size: will be the number of individual blocks which passed the check, with a minimum value of 1.
            i.e. if 4 blocks pass the check the return will be 4, but if there are only single blocks it will be 1.
            If the list passed is length 0 then height_count will return 0 and it will automatically pass successfully.
        """

        if check_z_height:
            # TODO(ahundt) Remove this call to self.get_camera_data. Added because the
            # valid_depth_heightmap here used for row checking is delayed by one action
            # Figure out why.
            valid_depth_heightmap, _, _, _, _, _ = self.get_camera_data(return_heightmaps=True)

            success, row_size = utils.check_row_success(valid_depth_heightmap, prev_z_height=prev_z_height)
            return success, row_size

        else:
            if len(object_color_sequence) < 1:
                print('check_row() object_color_sequence length is 0 or 1, so there is nothing to check and it passes automatically')
                return True, 1

            pos = np.asarray(self.get_obj_positions())
            posyx = copy.deepcopy(pos)
            posyx[:, [0,1]] = posyx[:, [1,0]]
            success = False
            row_size = 1
            row_length = len(object_color_sequence)
            # Color order of blocks doesn't matter, just the length of the sequence.
            # Therefore, check every row_length-size subset of blocks to see if
            # they are in a row and, if so, whether they are close enough
            # together.

            # lists all the possible subsets of blocks to check, for each possible length of row (except 1).
            # So for 3 objects, this would be:
            # [[[0,1], [0,2], [1,2]], [[0,1,2]]]
            all_block_indices = [map(list, itertools.combinations(np.arange(num_obj), length))
                                    for length in range(1, num_obj+1)]

            successful_block_indices = []
            for block_indices_of_length in all_block_indices:
                for block_indices in block_indices_of_length:
                    # check each rotation angle for a possible row
                    # print('checking {}'.format(block_indices))
                    specific_success, specific_row_size, specific_successful_block_indices = self.check_specific_blocks_for_row(pos, block_indices, distance_threshold, separation_threshold, object_color_sequence, row_size, success)
                    if specific_row_size > row_size:
                        success = specific_success
                        row_size = max(row_size, specific_row_size)
                        successful_block_indices = specific_successful_block_indices
                    else:
                        # TODO(ahundt) FIX HACK switch axis to yx order, to workaround the problem where it cannot check vertical lines for rows
                        specific_success, specific_row_size, specific_successful_block_indices = self.check_specific_blocks_for_row(posyx, block_indices, distance_threshold, separation_threshold, object_color_sequence, row_size, success)
                        if specific_row_size > row_size:
                            success = specific_success
                            row_size = max(row_size, specific_row_size)
                            successful_block_indices = specific_successful_block_indices


            print('check_row: {} | row_size: {} | blocks: {}'.format(
                success, row_size, np.array(self.color_names)[successful_block_indices]))
            return success, row_size

    def check_specific_blocks_for_row(self, pos, block_indices, distance_threshold, separation_threshold, object_color_sequence, row_size, success):
        """ check_row helper function to workaround that it cannot currently check vertical rows of blocks.
        """
        # TODO(ahundt) FIX HACK switch axis to yx order, to workaround the problem where it cannot check vertical lines for rows
        successful_block_indices = []
        xs = pos[block_indices][:, 0]
        ys = pos[block_indices][:, 1]
        if xs.size == 0 or ys.size == 0:
            # there is nothing to fit, 
            # not successful, row size 0, and empty block indices
            return False, 0, successful_block_indices
        # print('xs: {}'.format(xs))
        # print('ys: {}'.format(ys))
        m, b = utils.polyfit(xs, ys, 1)

        # print('m, b: {}, {}'.format(m, b))
        theta = np.arctan(m)  # TODO(bendkill): use arctan2?
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        T = np.array([0, -b, 0])
        # aligned_pos rotates X along the line of best fit (in x,y), so y should be small
        aligned_pos = np.array([np.matmul(R, p + T) for p in pos[block_indices]])

        aligned = True
        median_z = np.median(aligned_pos[:, 2])
        for p in aligned_pos:
            # print('distance from line: {:.03f}'.format(p[1]))
            if abs(p[1]) > distance_threshold or abs(p[2] - median_z) > distance_threshold:
                # too far from line on table, or blocks are not on the same Z plane
                aligned = False
                break

        indices = aligned_pos[:, 0].argsort()
        xs = aligned_pos[indices, 0]
        if aligned and utils.check_separation(xs, separation_threshold):
            # print('valid row along', theta, 'with indices', block_indices)
            if self.grasp_color_task:
                success = np.equal(indices, object_color_sequence).all()
            else:
                success = True
            successful_block_indices = block_indices
            row_size = max(len(block_indices), row_size)
        return success, row_size, successful_block_indices

    def check_stack(self, object_color_sequence, crop_stack_sequence=True,
            horiz_distance_threshold=0.06, vert_distance_threshold=0.06, top_idx=-1,
            pos=None, return_inds=False):
        """ Check for a complete stack in the correct order from bottom to top.

        Input: vector length of 1, 2, or 3
        Example: goal = [0] or [0,1] or [0,1,3]

        # Arguments

        object_color_sequence: vector indicating the index order of self.object_handles we expect to grasp.
        horiz_distance_threshold: The max distance cutoff between blocks(horizontal) in meters for the stack to be considered complete.
        vert_distance_threshold: The max distance cutoff between blocks(vertical) in meters for the stack to be considered complete.


        # Returns

        List [success, height_count].
        success: will be True if the stack matches the specified order from bottom to top, False otherwise.
        height_count: will be the number of individual blocks which passed the check, with a minimum value of 1.
            i.e. if 4 blocks pass the check the return will be 4, but if there are only single blocks it will be 1.
            If the list passed is length 0 then height_count will return 0 and it will automatically pass successfully.
        """
        # TODO(ahundt) support the check after a specific grasp in case of successful grasp topple. Perhaps allow the top block to be specified?
        checks = len(object_color_sequence) - 1
        if checks <= 0:
            print('check_stack() object_color_sequence length is 0 or 1, so there is nothing to check and it passes automatically')
            return True, checks+1

        # TODO(killeen) move grasp_color_task check to end, want to find stacks even if the order isn't right.

        # if block positions aren't specified, call get_obj_positions
        if pos is None:
            print('regenerating pos')
            pos = np.asarray(self.get_obj_positions())

        # Assume the stack will work out successfully
        # in the end until proven otherwise
        goal_success = True
        if not self.grasp_color_task:
            # Automatically determine the color order.
            # We don't worry about the colors, just the length of the sequence.
            # This should even handle 2 stacks of 2 blocks after a single place success
            # TODO(ahundt) See if there are any special failure cases common enough to warrant more code improvements
            num_obj = len(object_color_sequence)
            # object_z_positions = np.array(pos[:,2])
            # object_color_sequence = object_z_positions.argsort()[:num_obj][::-1]
            # object indices sorted highest to lowest
            # low2high_idx = object_z_positions.argsort()
            low2high_idx = np.array(pos[:, 2]).argsort()
            high_idx = low2high_idx[top_idx]
            low2high_pos = pos[low2high_idx, :]
            # filter objects closest to the highest block in x, y based on the threshold
            # ordered from low to high, boolean mask array
            nearby_obj = np.linalg.norm(low2high_pos[:, :2] - pos[high_idx, :2], axis=1) < \
                    (horiz_distance_threshold/2)
            # print('nearby:', nearby_obj)
            # take num_obj that are close enough from bottom to top
            # TODO(ahundt) auto-generated object_color_sequence definitely has some special case failures, check if it is good enough
            object_color_sequence = low2high_idx[nearby_obj]
            if len(object_color_sequence) < num_obj:
                print('check_stack() False, not enough nearby objects for a successful stack! ' \
                        'expected at least ' + str(num_obj) + ' nearby objects, but only counted: ' + \
                        str(len(object_color_sequence)))
                # there aren't enough nearby objects for a successful stack!
                checks = len(object_color_sequence) - 1
                # We know the goal won't be met, so goal_success is False
                # But we still need to count the actual stack height so set the variable for later
                goal_success = False
            elif crop_stack_sequence:
                # cut out objects we don't need to check if crop_stack_sequence is set
                object_color_sequence = object_color_sequence[:num_obj+1]
            else:
                checks = len(object_color_sequence) - 1
            # print('auto object_color_sequence: ' + str(object_color_sequence))

        # print('bottom: ' + str(object_color_sequence[:-1]))
        # print('top: ' + str(object_color_sequence[1:]))
        idx = 0
        for idx in range(checks):
            bottom_pos = pos[object_color_sequence[idx]]
            top_pos = pos[object_color_sequence[idx+1]]
            # Check that Z is higher by at least half the distance threshold
            # print('bottom_pos:', bottom_pos)
            # print('top_pos:', top_pos)
            # print('distance_threshold: ', distance_threshold)
            if top_pos[2] < (bottom_pos[2] + vert_distance_threshold / 2.0):
                print('check_stack(): not high enough for idx: ' + str(idx))
                if return_inds:
                    return False, idx + 1, object_color_sequence

                return False, idx + 1

            # Check that the blocks are near each other
            dist = np.linalg.norm(np.array(bottom_pos) - np.array(top_pos))
            # print('distance: ', dist)
            if dist > vert_distance_threshold:
                print('check_stack(): too far apart')
                if return_inds:
                    return False, idx + 1, object_color_sequence

                return False, idx + 1

        detected_height = min(idx + 2, len(object_color_sequence))
        print('check_stack() current detected stack height: ' + str(detected_height))

        if return_inds:
            return goal_success, detected_height, object_color_sequence

        return goal_success, detected_height

    def vertical_square_partial_success(self, current_stack_goal, check_z_height,
            row_dist_thresh=0.02, separation_threshold=0.1, stack_dist_thresh=0.06):
        """
            Checks if the last action was successful.
            NOTE: could be bugs with this, need to use location of last action to be sure
            current_stack_goal: array with current goal (length represents goal structure size)
            Returns:
                success: whether the previous action was successful
                structure_progress: step of task we are on (e.g. 2 if structure has 2 blocks)
        """

        # initially no stacks or rows, 1 block in structure
        num_stacks = 0
        has_row = False
        structure_size = 1

        if check_z_height:
            raise NotImplementedError

        # partial success is true if we meet the current stack goal (# of blocks in structure)

        # get object positions (array with each object position)
        pos = np.asarray(self.get_obj_positions())

        # sort indices of blocks by z value
        low2high_idx = np.array(pos[:, 2]).argsort()

        # check if we are currently holding a block, then we need to use -2 for top_idx
        if pos[low2high_idx[-1], 2] > 0.2:
            top_idx = -2
        else:
            top_idx = -1

        # first check for any stacks
        # NOTE if there isn't a first stack, then all blocks must be on the table so
        # we don't need to check for a second stack

        # for first stack, check if the highest block forms a stack, make sure to store inds of blocks in stack
        # top_idx is set to the index of low2high_idx we want to check the stack at
        _, stack_height, first_stack_inds = self.check_stack(np.ones(2), crop_stack_sequence=False,
                top_idx=top_idx, horiz_distance_threshold=stack_dist_thresh, pos=pos, return_inds=True)
        second_stack_inds = None

        if stack_height > 1:
            # we have at least 1 stack
            num_stacks += 1

            # iterate through all blocks except blocks in first stack to check for another stack
            for i, block_ind in enumerate(low2high_idx[::-1]):
                # skip blocks in first stack
                if block_ind in first_stack_inds:
                    continue

                # check for 2nd stack (use index of block_ind in low2high_idx)
                top_idx = len(low2high_idx) - 1 - i
                _, stack_height, second_stack_inds = self.check_stack(np.ones(2),
                        crop_stack_sequence=False, top_idx=top_idx, pos=pos, return_inds=True,
                        horiz_distance_threshold=stack_dist_thresh)

                if stack_height > 1:
                    num_stacks += 1
                    break

        # now check for rows

        # if we have 2 stacks, check the bottom blocks of each stack
        if num_stacks == 2:
            lowest_blocks = np.array([first_stack_inds[0], second_stack_inds[0]]).astype(int)
            has_row, _, _ = self.check_specific_blocks_for_row(pos, lowest_blocks,
                    row_dist_thresh, separation_threshold, None, 1, False)

            if has_row:
                # we have 2 stacks and they form a row (structure is complete)
                structure_size = 4
            else:
                structure_size = 2

        # if we have 1 stack, check bottom block of stack with all other blocks
        elif num_stacks == 1:
            # structure_size is at least 2
            structure_size = 2
            lowest_block = first_stack_inds[0]
            for block_ind in low2high_idx:
                # skip blocks already in stack
                if block_ind in first_stack_inds: continue

                lowest_blocks = np.array([lowest_block, block_ind]).astype(int)
                has_row, _, _ = self.check_specific_blocks_for_row(pos, lowest_blocks,
                        row_dist_thresh, separation_threshold, None, 1, False)

                if has_row:
                    # there is 1 stack and 1 row
                    structure_size = 3
                    break

        # if we have 0 stacks, check all pairs of blocks
        else:
            for i in range(len(low2high_idx)):
                for j in range(i+1, len(low2high_idx)):
                    lowest_blocks = np.array([low2high_idx[i], low2high_idx[j]]).astype(int)
                    has_row, _, _ = self.check_specific_blocks_for_row(pos, lowest_blocks,
                            row_dist_thresh, separation_threshold, None, 1, False)

                    if has_row:
                        structure_size = 2
                        break

                if has_row:
                    break

        print("vertical square partial success: structure height:", structure_size, "has_row:",
                has_row, "num stacks:", num_stacks)

        # success if we match or exceed current stack goal, also return structure size
        return structure_size >= len(current_stack_goal), structure_size

    def unstacking_partial_success(self, prev_structure_progress, distance_threshold=0.06, top_idx=-1, check_z_height=False, depth_img=None):
        """ Check stack height, set partial_stack_success flag to true if stack height decreases on grasp

        # Arguments

        prev_stack_height: height of stack before last action was taken
        distance_threshold: The max distance cutoff between blocks in meters for the stack to be considered complete.

        # Returns

        List [success, stack_height].
        success: will be True if the stack matches the specified order from bottom to top, False otherwise.
        stack_height: number of blocks in stack
        """
        if check_z_height:
            _, stack_height, _ = self.check_z_height(depth_img, prev_structure_progress)
            stack_height = int(np.rint(stack_height))
        else:
            # get object positions (array with each object position)
            pos = np.asarray(self.get_obj_positions())

            # sort indices of blocks by z value
            low2high_idx = np.array(pos[:, 2]).argsort()

            # check if we are currently holding a block, then we need to use -2 for top_idx
            if pos[low2high_idx[-1], 2] > 0.2:
                top_idx = -2

            # run check stack to get height of stack
            _, stack_height, _ = self.check_stack(np.ones(4), pos=pos, top_idx=top_idx, horiz_distance_threshold=distance_threshold)

        # structure progress is 1 when stack is full, 2 when we unstack 1 block, and so on
        structure_progress = 5 - stack_height

        # check if we decreased or maintained last stack height
        goal_success = (structure_progress >= prev_structure_progress)
        print('unstacking_partial_success() structure_progress:', structure_progress,
                'prev_structure_progress:', prev_structure_progress, 'goal_success:',
                goal_success)

        return goal_success, structure_progress

    def manual_progress_check(self, prev_structure_progress, task_type):
        while True:
            try:
                progress = float(input(" ".join(["For task", task_type.upper(),
                    "input current structure size: "])))
                break
            except ValueError:
                print("ENTER AN INTEGER!!!!")
                continue

        if progress < prev_structure_progress:
            needed_to_reset = True
        else:
            needed_to_reset = False

        if progress > prev_structure_progress:
            stack_matches_goal = True
        elif task_type == 'unstack' and progress >= prev_structure_progress:
            stack_matches_goal = True
        else:
            stack_matches_goal = False

        return stack_matches_goal, progress, needed_to_reset

    def check_incremental_height(self, input_img, current_stack_goal):
        goal_success = False
        goal, max_z, decrease_threshold = self.check_z_height(input_img)
        #TODO(hkwon214) Double check this
        current_stack_goal = len(current_stack_goal)
        if (max_z <= 0.069):
            detected_height = 1
        elif (max_z > 0.069) and (max_z <= 0.11):
            detected_height = 2
        elif (max_z > 0.11) and (max_z <= 0.156):
            detected_height = 3
        # elif (max_z > 0.156) and (max_z <= 0.21):
        #     detected_height = 4
        else:
            detected_height = 4
        #TODO(hkwon214) What happens if the height is above the limit?
        if current_stack_goal == detected_height:
            goal_success = True
        return goal_success, detected_height

    def check_z_height(self, input_img, prev_height=0.0, increase_threshold=0.03, decrease_threshold=0.02, reward_multiplier=None):
        """ Checks the maximum z height after applying a median filter. Includes checks for significant increases and decreases.

        # Returns

            [goal_success, max_z, needed_to_reset]

            goal_success: has height increased by the increase_threshold
            max_z: what is the current maximum z height in the image
            neede_to_reset: has the height decreased from the prev_height enough to warrant special reset/recovery actions.
            reward_multiplier: Converts scale from "meters" to "blocks" scale. Default None means 20.0 if self.is_sim else 22.0.
        """
        if reward_multiplier is None:
            reward_multiplier = 20.0 if self.is_sim else 22.0
        # TODO(ahundt) make reward multiplier, increase threshold, and decrease threshold command line parameters which can be modified.
        img_median = ndimage.median_filter(input_img, size=5)
        max_z = np.max(img_median) * reward_multiplier
        # TODO(ahundt) should the reward multiplier be applied to increase_threshold, or should the parameter be totally indepenedent?
        goal_success = max_z >= (prev_height + (increase_threshold * reward_multiplier))
        needed_to_reset = False
        max_workspace_height = prev_height - (decrease_threshold * reward_multiplier)
        if decrease_threshold is not None and max_z < max_workspace_height:
            needed_to_reset = True
        print('prev_height: ' + str(prev_height) + ' max_z: '  + str(max_z) + \
              ' goal_success: ' + str(goal_success) + ' needed to reset: ' + \
              str(needed_to_reset) + ' max_workspace_height: ' + str(max_workspace_height) + ' <<<<<<<<<<<')
        return goal_success, max_z, needed_to_reset

    def restart_real(self):
        # reset objects for stacking
        if self.place_task:
            return self.reposition_objects()

        # reset objects for pushing and grasping
        # position just over the box to dump
        # [0.2035772  0.14621875 0.07735696]
        # Compute tool orientation from heightmap rotation angle
        self.open_gripper()
        tool_orientation = [0, np.pi, 0.0] # Real Good Robot
        above_bin_waypoint = [ 0.6, -0.1, 0.25]
        self.move_to(above_bin_waypoint, tool_orientation)
        time.sleep(.1)
        above_box_waypoint = [0.32, 0.29,  0.30]
        self.move_to(above_box_waypoint, tool_orientation)
        time.sleep(.1)
        # grasp_box is to grasp the edge of of the box
        grasp_box =  [0.32, 0.29, 0.22]
        self.move_to(grasp_box, tool_orientation)
        time.sleep(.1)
        self.close_gripper()
        drag_goal = [0.67, -0.16, 0.20]
        self.move_to(drag_goal, tool_orientation)
        time.sleep(.1)
        move_up_goal = [0.67, -0.16, 0.33]
        self.move_to(move_up_goal, tool_orientation)
        time.sleep(.1)
        self.move_to(above_box_waypoint, tool_orientation)
        time.sleep(.1)
        self.move_to(grasp_box, tool_orientation)
        time.sleep(.1)
        self.open_gripper()
        self.move_to(above_box_waypoint, tool_orientation)
        time.sleep(.1)
        self.move_to(above_bin_waypoint, tool_orientation)
        time.sleep(.1)
        return self.go_home(block_until_home=True)

    def shutdown(self):
        if self.is_sim:
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxFinish(-1)
