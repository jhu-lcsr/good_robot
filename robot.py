import socket
import select
import struct
import time
import os
import numpy as np
import itertools
import utils
from simulation import vrep
from scipy import ndimage, misc
try:
    from gripper.robotiq_2f_gripper_ctrl import RobotiqCGripper
except ImportError:
    print('Real robotiq gripper control is not available. '
          'Ensure pymodbus is installed:\n'
          '    pip3 install --user --upgrade pymodbus\n')
    RobotiqCGripper = None


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
                 is_testing=False, test_preset_cases=None, test_preset_file=None, place=False, grasp_color_task=False,
                 real_gripper_ip='192.168.1.11', calibrate=False):
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
        self.place_task = place
        self.grasp_color_task = grasp_color_task
        self.sim_home_position = [-0.3, 0.0, 0.45]
        # self.gripper_ee_offset = 0.17
        self.gripper_ee_offset = 0.15
        self.background_heightmap = None

        # HK: If grasping specific block color...
        #
        # TODO: Change to random color not just red block using  (b = [0, 1, 2, 3] np.random.shuffle(b)))
        # after grasping, put the block back
        self.color_names = ['blue', 'green', 'yellow', 'red', 'brown', 'orange', 'gray', 'purple', 'cyan', 'pink']

        # If in simulation...
        if self.is_sim:
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
            self.mesh_list = os.listdir(self.obj_mesh_dir)

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

            if tcp_port == 30002:
                print("WARNING: default tcp port changed to 19997 for is_sim")
                tcp_port = 19997

            # Connect to simulator
            vrep.simxFinish(-1) # Just in case, close all opened connections
            self.sim_client = vrep.simxStart('127.0.0.1', tcp_port, True, True, 5000, 5) # Connect to V-REP on port 19997
            if self.sim_client == -1:
                print('Failed to connect to simulation (V-REP remote API server). Exiting.')
                exit()
            else:
                print('Connected to simulation.')
                self.restart_sim()

            self.is_testing = is_testing
            self.test_preset_cases = test_preset_cases
            self.test_preset_file = test_preset_file

            # Setup virtual camera in simulation
            self.setup_sim_camera()

            # If testing, read object meshes and poses from test case file
            if self.is_testing and self.test_preset_cases:
                self.load_preset_case()

            # Add objects to simulation environment
            self.add_objects()


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
            self.joint_acc = 5.0 # Safe: 1.4  Fast: 8
            self.joint_vel = 3.0 # Safe: 1.05  Fast: 3

            # Joint tolerance for blocking calls
            self.joint_tolerance = 0.01

            # Default tool speed configuration
            self.tool_acc = 1.2 # Safe: 0.5 Fast: 1.2
            self.tool_vel = 0.5 # Safe: 0.2 Fast: 0.5
            self.move_sleep = 1.0 # Safe: 2.0 Fast: 1.0

            # Tool pose tolerance for blocking calls
            self.tool_pose_tolerance = [0.002,0.002,0.002,0.01,0.01,0.01]

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
            if os.path.isfile('real/robot_base_to_camera_pose.txt') and os.path.isfile('real/camera_depth_scale.txt'):
                self.cam_pose = np.loadtxt('real/robot_base_to_camera_pose.txt', delimiter=' ')
                self.cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')
            else:
                print('WARNING: Camera Calibration is not yet available, running calibrate.py '
                      'will create the required files: real/robot_base_to_camera_pose.txt and real/camera_depth_scale.txt')
                # Camera calibration
                self.cam_pose = None
                self.cam_depth_scale = None

            if os.path.isfile('real/background_heightmap.depth.png'):
                import cv2
                 # load depth image saved in 1e-5 meter increments
                 # see logger.py save_heightmaps() and trainer.py load_sample()
                 # for the corresponding save and load functions
                self.background_heightmap = np.array(cv2.imread('real/background_heightmap.depth.png', cv2.IMREAD_ANYDEPTH)).astype(np.float32) / 100000

    def load_preset_case(self, test_preset_file=None):
        if test_preset_file is None:
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

        # Get handle to camera
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

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


    def reposition_object_at_list_index_randomly(self, list_index):
        object_handle = self.object_handles[list_index]
        self.reposition_object_randomly(object_handle)


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
            if failure_count > 10 or len(self.object_handles) > len(self.obj_mesh_ind):
                # If the simulation is not currently running, attempt to recover by restarting the simulation
                self.restart_sim()
                self.object_handles = []
                self.vrep_names = []
                self.object_colors = []
            for object_idx in range(len(self.obj_mesh_ind)):
                curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
                if self.is_testing and self.test_preset_cases:
                    curr_mesh_file = self.test_obj_mesh_files[object_idx]
                # TODO(ahundt) define more predictable object names for when the number of objects is beyond the number of colors
                curr_shape_name = 'shape_%02d' % object_idx
                self.vrep_names.append(curr_shape_name)
                drop_x, drop_y, object_position, object_orientation = self.generate_random_object_pose()
                if self.is_testing and self.test_preset_cases:
                    object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1], self.test_obj_positions[object_idx][2]]
                    object_orientation = [self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
                # Set the colors in order
                object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]
                # If there are more objects than total colors this line will break,
                # applies mod to loop back to the first color.
                object_color_name = self.color_names[object_idx % len(self.color_names)]
                # add the color of this object to the list.
                self.object_colors.append(object_color_name)
                print('Adding object: ' + curr_mesh_file + ' as ' + curr_shape_name)
                do_break = False
                ret_ints = []
                while len(ret_ints) == 0:
                    do_break = False
                    ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
                    if ret_resp == 8:
                        print('Failed to add ' + curr_mesh_file + ' to simulation. Auto retry ' + str(failure_count))
                        failure_count += 1
                        if failure_count % 5 == 4:
                            # If a few failures happen in a row, do a simulation reset and try again
                            do_break = True
                            break
                        elif failure_count > 50:
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


    def restart_sim(self):

        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        sim_ok = False
        while not sim_ok: # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(0.5)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(0.5)
            sim_started = vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking) > 0
            time.sleep(0.5)
            sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
            # check sim, but we are already in the restart loop so don't recurse
            sim_ok = sim_started and self.check_sim(restart_if_not_ok=False)


    def check_sim(self, restart_if_not_ok=True):
        # buffer_meters = 0.1  # original buffer value
        buffer_meters = 0.1
        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - buffer_meters and gripper_position[0] < self.workspace_limits[0][1] + buffer_meters and gripper_position[1] > self.workspace_limits[1][0] - buffer_meters and gripper_position[1] < self.workspace_limits[1][1] + buffer_meters and gripper_position[2] > self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if restart_if_not_ok and not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
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


    # def stop_sim(self):
    #     if self.is_sim:
    #         # Now send some data to V-REP in a non-blocking fashion:
    #         # vrep.simxAddStatusbarMessage(sim_client,'Hello V-REP!',vrep.simx_opmode_oneshot)

    #         # # Start the simulation
    #         # vrep.simxStartSimulation(sim_client,vrep.simx_opmode_oneshot_wait)

    #         # # Stop simulation:
    #         # vrep.simxStopSimulation(sim_client,vrep.simx_opmode_oneshot_wait)

    #         # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    #         vrep.simxGetPingTime(self.sim_client)

    #         # Now close the connection to V-REP:
    #         vrep.simxFinish(self.sim_client)


    def get_obj_positions(self):
        if not self.is_sim:
            raise NotImplementedError('get_obj_positions() only supported in simulation, if you are training stacking try specifying --check_z_height')
        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

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


    def reposition_objects(self, workspace_limits=None):

        if self.is_sim:
            # Move gripper out of the way
            success = self.move_to([-0.3, 0, 0.3], None)
            if not success:
                return False
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            # vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
            # time.sleep(1)

            for object_handle in self.object_handles:

                # Drop object at random x,y location and random orientation in robot workspace
                self.reposition_object_randomly(object_handle)
                time.sleep(0.5)
            # an extra half second so things settle down
            time.sleep(0.5)
            return True

        # TODO(ahundt) add real robot support for reposition_objects


    def get_camera_data(self):

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
            # Get color and depth image from ROS service
            color_img, depth_img = self.camera.get_data()
            depth_img = depth_img.astype(float) / 1000 # unit: mm -> meter
            # color_img = self.camera.color_data.copy()
            # depth_img = self.camera.depth_data.copy()

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
                gripper_fully_closed = self.gripper.is_closed()

        return gripper_fully_closed


    def open_gripper(self, nonblocking=False):
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
            while gripper_joint_position < 0.03: # Block until gripper is fully open
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
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


    def move_to(self, tool_position, tool_orientation=None, timeout_seconds=10, heightmap_rotation_angle=None):

        if self.is_sim:
            if tool_orientation is None and heightmap_rotation_angle is not None:
                # Compute tool orientation from heightmap rotation angle
                tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2
                # TODO(ahundt) bring in some of the code from grasp() here to correctly update the orientation
                raise NotImplementedError
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            if np.isnan(UR5_target_position).any():
                return False

            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.02*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_magnitude/0.02))

            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
            return True
        else:
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
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "set_tcp(p[-0.1,0.0,0.0,0.0,0.0,0.0])\n"
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (tool_position[0],tool_position[1],tool_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.tool_acc,self.tool_vel)
            print(tcp_command)
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()

            # Block until robot reaches target tool position
            return self.block_until_cartesian_position(tool_position, timeout_seconds=timeout_seconds)

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


    def move_joints(self, joint_configuration):
        if self.is_sim:
            raise NotImplementedError

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
        while not all([np.abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
            state_data = self.get_state()
            actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
            time.sleep(0.01)


    def go_home(self, block_until_home=False):
        if self.is_sim:
            return self.move_to(self.sim_home_position, None)
        else:
            self.move_joints(self.home_joint_config)
            if block_until_home:
                return self.block_until_home()
            else:
                return True


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

    # Primitives ----------------------------------------------------------

    def grasp(self, position, heightmap_rotation_angle, object_color=None, workspace_limits=None, go_home=True):
        """
        object_color: The index in the list self.color_names expected for the object to be grasped. If object_color is None the color does not matter.
        """
        if workspace_limits is None:
            workspace_limits = self.workspace_limits
        print('Executing: grasp at (%f, %f, %f) orientation: %f' % (position[0], position[1], position[2], heightmap_rotation_angle))

        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

            # Avoid collision with floor
            position = np.asarray(position).copy()
            position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)

            # Move gripper to location above grasp target, this is the pre-grasp and post-grasp height
            # grasp_location_margin = 0.15
            grasp_location_margin = 0.2
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_grasp_target
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            # prevent division by 0, source: https://stackoverflow.com/a/37977222/99379
            move_step = 0.05 * np.divide(move_direction, move_magnitude, out=np.zeros_like(move_direction), where=move_magnitude!=0)
            # move_step = 0.05*move_direction/move_magnitude
            # print('move direction: ' + str(move_direction))
            # print('move step: ' + str(move_step))
            # TODO(ahundt) 0 steps may still be buggy in certain cases
            if move_step[0] == 0:
                num_move_steps = 0
            else:
                num_move_steps = int(np.floor(move_direction[0]/move_step[0]))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

            # Ensure gripper is open
            self.open_gripper()

            # Approach grasp target
            self.move_to(position, None)

            # Close gripper to grasp target
            gripper_full_closed = self.close_gripper()

            # Move gripper to location above grasp target
            self.move_to(location_above_grasp_target, None)
            # move to the simulator home position
            if go_home:
                self.go_home()

            # Check if grasp is successful
            gripper_full_closed = self.close_gripper()
            grasp_success = not gripper_full_closed

            # HK: Check if right color is grasped
            color_success = False
            if grasp_success and self.grasp_color_task:
                color_success = self.check_correct_color_grasped(object_color)
                print('Correct color was grasped: ' + str(color_success))

            # Move the grasped object elsewhere if place = false
            # if grasp_success and not self.place_task and self.grasp_color_task:
            #     object_positions = np.asarray(self.get_obj_positions())
            #     object_positions = object_positions[:,2]
            #     grasped_object_ind = np.argmax(object_positions)
            #     grasped_object_handle = self.object_handles[grasped_object_ind]
            #     # TODO: HK: check if any block is grasped and put it back at a random place
            #     color = np.array([0, 0, 0, 0, 0,  0,1, 0, 0, 0]) # red block
            #     color_index = np.where(color==1)
            #     vrep.simxSetObjectPosition(self.sim_client,grasped_object_handle,-1,(-0.5, 0.5 + 0.05*float(grasped_object_ind), 0.1),vrep.simx_opmode_blocking)
            #     if grasped_object_ind == color_index[0]:
            #         workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
            #         drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
            #         drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
            #         object_position = [drop_x, drop_y, 0.15]
            #         object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            #         vrep.simxSetObjectPosition(self.sim_client, grasped_object_handle, -1, object_position, vrep.simx_opmode_blocking)
            #         vrep.simxSetObjectOrientation(self.sim_client, grasped_object_handle, -1, object_orientation, vrep.simx_opmode_blocking)

            # HK: Place grasped object at a random place
            if grasp_success:
                if self.place_task:
                    return grasp_success, color_success
                if not self.place_task and object_color is not None:
                    high_obj_list_index, high_obj_handle = self.get_highest_object_list_index_and_handle()
                    self.reposition_object_randomly(high_obj_handle)
                    # TODO: HK: check if any block is grasped and put it back at a random place
                    # color = np.array([0, 0, 0, 0, 0,  0,1, 0, 0, 0]) # red block
                    # color_index = np.where(color==1)
                    # vrep.simxSetObjectPosition(self.sim_client,grasped_object_handle,-1,(-0.5, 0.5 + 0.05*float(grasped_object_ind), 0.1),vrep.simx_opmode_blocking)
                    #     workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])

                # HK: Original Method
                elif not self.place_task and not self.grasp_color_task:
                    object_positions = np.asarray(self.get_obj_positions())
                    object_positions = object_positions[:,2]
                    grasped_object_ind = np.argmax(object_positions)
                    grasped_object_handle = self.object_handles[grasped_object_ind]
                    vrep.simxSetObjectPosition(self.sim_client,grasped_object_handle,-1,(-0.5, 0.5 + 0.05*float(grasped_object_ind), 0.1),vrep.simx_opmode_blocking)
                else:
                    raise NotImplementedError(
                        'grasp() call specified a task which does not match color specific stacking or grasp+pushing... '
                        'this is a bug or is not yet implemented, you will need to make a code change or run the program with different settings.')

        else:
            # Warning: "Real Good Robot!" specific hack, increase gripper height for our different mounting config
            position[2] += self.gripper_ee_offset - 0.01
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
            position = np.asarray(position).copy()

            # find position halfway between the current and final.
            print("Grasp position before applying workspace bounds: " + str(position))
            position[2] = max(position[2], workspace_limits[2][0] + self.gripper_ee_offset + 0.04)
            position[2] = min(position[2], workspace_limits[2][1] + self.gripper_ee_offset - 0.01)
            up_pos = np.array([position[0],position[1],position[2]+0.1])

            print("Real Good Robot grasping at: " + str(position) + ", " + str(tool_orientation))
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "def process():\n"
            tcp_command += " set_digital_out(8,False)\n"
            if go_home:
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (self.home_cart_low[0],self.home_cart_low[1],self.home_cart_low[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc,self.joint_vel)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (up_pos[0],up_pos[1],up_pos[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc,self.joint_vel)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (position[0],position[1],position[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc*0.1,self.joint_vel*0.1)
            tcp_command += " set_digital_out(8,True)\n"
            tcp_command += "end\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()

            self.block_until_cartesian_position(position)
            time.sleep(0.1)
            gripper_fully_closed = self.close_gripper()
            color_success = None
            bin_position = np.array([0.33, 0.40, 0.33 + self.gripper_ee_offset])

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

            if go_home:
                self.go_home()
            else:
                # go back to the grasp up pos
                self.move_to(up_pos,[tool_orientation[0],tool_orientation[1],0.0])

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
        print('Real Robot push at (%f, %f, %f) angle: %f' % (position[0], position[1], position[2], heightmap_rotation_angle))

        if self.is_sim:

            # Compute tool orientation from heightmap rotation angle
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

            # Adjust pushing point to be on tip of finger
            position[2] = position[2] + 0.026

            # Compute pushing direction
            push_orientation = [1.0,0.0]
            push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle)])

            # Move gripper to location above pushing point
            pushing_point_margin = 0.1
            location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

            # Compute gripper position and linear movement increments
            tool_position = location_above_pushing_point
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05*move_direction/move_magnitude
            num_move_steps = int(np.floor(move_direction[0]/move_step[0]))

            # Compute gripper orientation and rotation increments
            sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
            rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
            num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotation_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

            # Ensure gripper is closed
            self.close_gripper()

            # Approach pushing point
            self.move_to(position, None)

            # Compute target location (push to the right)
            push_length = 0.1
            target_x = min(max(position[0] + push_direction[0]*push_length, workspace_limits[0][0]), workspace_limits[0][1])
            target_y = min(max(position[1] + push_direction[1]*push_length, workspace_limits[1][0]), workspace_limits[1][1])
            push_length = np.sqrt(np.power(target_x-position[0],2)+np.power(target_y-position[1],2))

            # Move in pushing direction towards target location
            self.move_to([target_x, target_y, position[2]], None)

            # Move gripper to location above grasp target
            self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

            # move to the simulator home position
            if go_home:
                self.go_home()

            push_success = True

        else:
            # Warning: "Real Good Robot!" specific hack, increase gripper height for our different mounting config
            position[2] += self.gripper_ee_offset + 0.01

            # Compute tool orientation from heightmap rotation angle
            push_orientation = [1.0,0.0]
            tool_rotation_angle = heightmap_rotation_angle/2
            tool_orientation = np.asarray([push_orientation[0]*np.cos(tool_rotation_angle) - push_orientation[1]*np.sin(tool_rotation_angle), push_orientation[0]*np.sin(tool_rotation_angle) + push_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
            tool_orientation_angle = np.linalg.norm(tool_orientation)
            tool_orientation_axis = tool_orientation/tool_orientation_angle
            tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]

            # Compute push direction and endpoint (push to right of rotated heightmap)
            push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle), 0.0])
            target_x = min(max(position[0] + push_direction[0]*0.1, workspace_limits[0][0]), workspace_limits[0][1])
            target_y = min(max(position[1] + push_direction[1]*0.1, workspace_limits[1][0]), workspace_limits[1][1])
            push_endpoint = np.asarray([target_x, target_y, position[2]])
            push_direction.shape = (3,1)

            # Compute tilted tool orientation during push
            tilt_axis = np.dot(utils.euler2rotm(np.asarray([0,0,np.pi/2]))[:3,:3], push_direction)
            tilt_rotm = utils.angle2rotm(-np.pi/8, tilt_axis, point=None)[:3,:3]
            tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
            tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
            tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])

            # Push only within workspace limits
            position = np.asarray(position).copy()
            position[0] = min(max(position[0], workspace_limits[0][0]), workspace_limits[0][1])
            position[1] = min(max(position[1], workspace_limits[1][0]), workspace_limits[1][1])
            position[2] = max(position[2] + 0.005, workspace_limits[2][0] + 0.005) # Add buffer to surface
            up_pos = np.array([position[0],position[1],position[2]+0.1])

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
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.03)\n" % (position[0],position[1],position[2]+0.1,tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
            if go_home:
                tcp_command += " movej([%f" % self.home_joint_config[0]
                for joint_idx in range(1,6):
                    tcp_command = tcp_command + (",%f" % self.home_joint_config[joint_idx])
                tcp_command = tcp_command + "],a=%f,v=%f)\n" % (self.joint_acc, self.joint_vel)
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
            self.open_gripper(nonblocking=True)
            # time.sleep(0.25)

        return push_success

    def get_cartesian_position(self):
        if self.is_sim:
            raise NotImplementedError

        state_data = self.get_state()
        actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
        return actual_tool_pose

    def block_until_home(self, timeout_seconds=10):

        if self.is_sim:
            raise NotImplementedError
        # Don't wait for more than 20 seconds
        iteration_time_0 = time.time()
        while True:
            time_elapsed = time.time()-iteration_time_0
            if int(time_elapsed) > timeout_seconds:
                print('Move to Home Position Failed')
                return False
            state_data = self.get_state()
            actual_joint_pose = self.parse_tcp_state_data(state_data, 'joint_data')
            if all([np.abs(actual_joint_pose[j] - self.home_joint_config[j]) < self.joint_tolerance for j in range(5)]):
                print('Move to Home Position Complete')
                return True
            time.sleep(0.1)

    def block_until_cartesian_position(self, position, timeout_seconds=10):
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

    def block_until_joint_position(self, position, timeout_seconds=10):

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


    def place(self, position, heightmap_rotation_angle, workspace_limits=None, distance_threshold=0.06, go_home=True):
        """ Place an object, currently only tested for blocks.

        When in sim mode it assumes the current position of the robot and grasped object is higher than any other object.
        """
        if workspace_limits is None:
            workspace_limits = self.workspace_limits
        print('Executing: place at (%f, %f, %f)' % (position[0], position[1], position[2]))

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
            tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

            # Avoid collision with floor
            position[2] = max(position[2] + 0.04 + 0.02, workspace_limits[2][0] + 0.02)

            # Move gripper to location above place target
            place_location_margin = 0.1
            sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            location_above_place_target = (position[0], position[1], position[2] + place_location_margin)
            self.move_to(location_above_place_target, None)

            sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
            if tool_rotation_angle - gripper_orientation[1] > 0:
                increment = 0.2
            else:
                increment = -0.2
            while abs(tool_rotation_angle - gripper_orientation[1]) >= 0.2:
                vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + increment, np.pi/2), vrep.simx_opmode_blocking)
                time.sleep(0.005)
                sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

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
            return place_success
            #if abs(current_obj_z_location - position[2]) < 0.009:
            # if ((current_obj_z_location+distance_threshold/2) >= position[2]) and ((current_obj_z_location+distance_threshold/2) < (position[2]+distance_threshold)):
            #     place_success = True
            # else:
            #     place_success = False
            # return place_success
        else:

            # Warning: "Real Good Robot!" specific hack, increase gripper height for our different mounting config
            position[2] += self.gripper_ee_offset + 0.05
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
            position[2] = max(position[2], workspace_limits[2][0] + self.gripper_ee_offset + 0.04)
            position[2] = min(position[2], workspace_limits[2][1] + self.gripper_ee_offset - 0.01)
            up_pos = np.array([position[0],position[1],position[2]+0.1])

            print("Real Good Robot placing at: " + str(position) + ", " + str(tool_orientation))
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "def process():\n"
            tcp_command += " set_digital_out(8,False)\n"
            if go_home:
                tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (self.home_cart_low[0],self.home_cart_low[1],self.home_cart_low[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc,self.joint_vel)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (up_pos[0],up_pos[1],up_pos[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc,self.joint_vel)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (position[0],position[1],position[2],tool_orientation[0],tool_orientation[1],0.0,self.joint_acc*0.1,self.joint_vel*0.1)
            tcp_command += " set_digital_out(8,True)\n"
            tcp_command += "end\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()
            time.sleep(0.1)
            self.block_until_cartesian_position(position)
            time.sleep(0.1)

            self.open_gripper(nonblocking=True)
            move_to_result = self.move_to(up_pos)
            if go_home:
                # TODO(ahundt) save previous and new depth image, and if the depth at the place coordinate increased, return True for place success
                return self.go_home(block_until_home=True)
            else:
                return move_to_result


    def check_row(self, object_color_sequence,
                  num_obj=4,
                  distance_threshold=0.02,
                  separation_threshold=0.1,
                  num_directions=64):
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
        if len(object_color_sequence) < 1:
            print('check_row() object_color_sequence length is 0 or 1, so there is nothing to check and it passes automatically')
            return True, 1

        pos = np.asarray(self.get_obj_positions())
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
                xs = pos[block_indices][:, 0]
                ys = pos[block_indices][:, 1]
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
                    continue

        print('check_row: {} | row_size: {} | blocks: {}'.format(
            success, row_size, np.array(self.color_names)[successful_block_indices]))
        return success, row_size


    def check_stack(self, object_color_sequence, distance_threshold=0.06, top_idx=-1):
        """ Check for a complete stack in the correct order from bottom to top.

        Input: vector length of 1, 2, or 3
        Example: goal = [0] or [0,1] or [0,1,3]

        # Arguments

        object_color_sequence: vector indicating the index order of self.object_handles we expect to grasp.
        distance_threshold: The max distance cutoff between blocks in meters for the stack to be considered complete.


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
            nearby_obj = np.linalg.norm(low2high_pos[:, :2] - pos[high_idx][:2], axis=1) < (distance_threshold/2)
            # print('nearby:', nearby_obj)
            # take num_obj that are close enough from bottom to top
            # TODO(ahundt) auto-generated object_color_sequence definitely has some special case failures, check if it is good enough
            object_color_sequence = low2high_idx[nearby_obj]
            if len(object_color_sequence) < num_obj:
                print('check_stack() False, not enough nearby objects for a successful stack! '
                      'expected at least ' + str(num_obj) +
                      ' nearby objects, but only counted: ' + str(len(object_color_sequence)))
                # there aren't enough nearby objects for a successful stack!
                checks = len(object_color_sequence) - 1
                # We know the goal won't be met, so goal_success is False
                # But we still need to count the actual stack height so set the variable for later
                goal_success = False
                # TODO(ahundt) BUG this may actually return 1 when there is a stack of size 2 present, but 3 objects are needed
                # return False, 1
            else:
                # cut out objects we don't need to check
                object_color_sequence = object_color_sequence[:num_obj+1]
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
            if top_pos[2] < (bottom_pos[2] + distance_threshold / 2.0):
                print('check_stack(): not high enough for idx: ' + str(idx))
                return False, idx + 1
            # Check that the blocks are near each other
            dist = np.linalg.norm(np.array(bottom_pos) - np.array(top_pos))
            # print('distance: ', dist)
            if dist > distance_threshold:
                print('check_stack(): too far apart')
                return False, idx + 1
        detected_height = min(idx + 2, len(object_color_sequence))
        print('check_stack() current detected stack height: ' + str(detected_height))
        # TODO(ahundt) add check_stack for real robot
        return goal_success, detected_height

    def check_incremental_height(self,input_img, current_stack_goal):
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

    def check_z_height(self, input_img, prev_height=0.0, increase_threshold=0.03, decrease_threshold=0.02, reward_multiplier=20.0):
        """ Checks the maximum z height after applying a median filter. Includes checks for significant increases and decreases.

        # Returns

            [goal_success, max_z, needed_to_reset]

            goal_success: has height increased by the increase_threshold
            max_z: what is the current maximum z height in the image
            neede_to_reset: has the height decreased from the prev_height enough to warrant special reset/recovery actions.
        """
        # TODO(ahundt) make reward multiplier, increase threshold, and decrease threshold command line parameters which can be modified.
        img_median = ndimage.median_filter(input_img, size=5)
        max_z = np.max(img_median) * reward_multiplier
        # TODO(ahundt) should the reward multiplier be applied to increase_threshold, or should the parameter be totally indepenedent?
        goal_success = max_z >= (prev_height + (increase_threshold * reward_multiplier))
        needed_to_reset = False
        max_workspace_height = prev_height - decrease_threshold
        if decrease_threshold is not None and max_z < max_workspace_height:
            needed_to_reset = True
        print('prev_height: ' + str(prev_height) + ' max_z: '  + str(max_z) +
              ' goal_success: ' + str(goal_success) + ' needed to reset: ' + str(needed_to_reset) + ' max_workspace_height: ' + str(max_workspace_height) + ' <<<<<<<<<<<')
        return goal_success, max_z, needed_to_reset

    def restart_real(self):
        # position just over the box to dump
        # [0.2035772  0.14621875 0.07735696]
        # Compute tool orientation from heightmap rotation angle
        self.open_gripper()
        tool_orientation = [0, np.pi, 0.0] # Real Good Robot
        above_bin_waypoint = [ 0.6, -0.1, 0.25 + self.gripper_ee_offset]
        self.move_to(above_bin_waypoint, tool_orientation)
        time.sleep(.1)
        above_box_waypoint = [0.32, 0.29,  0.30+self.gripper_ee_offset]
        self.move_to(above_box_waypoint, tool_orientation)
        time.sleep(.1)
        # grasp_box is to grasp the edge of of the box
        grasp_box =  [0.32, 0.29, 0.22+self.gripper_ee_offset]
        self.move_to(grasp_box, tool_orientation)
        time.sleep(.1)
        self.close_gripper()
        drag_goal = [0.67, -0.16, 0.20+self.gripper_ee_offset]
        self.move_to(drag_goal, tool_orientation)
        time.sleep(.1)
        move_up_goal = [0.67, -0.16, 0.36+self.gripper_ee_offset]
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
        self.go_home(block_until_home=True)
        # tool_rotation_angle = -np.pi/4
        # tool_rotation_angle = np.pi
        # tool_orientation = np.asarray([grasp_orientation[0]*np.cos(tool_rotation_angle) - grasp_orientation[1]*np.sin(tool_rotation_angle), grasp_orientation[0]*np.sin(tool_rotation_angle) + grasp_orientation[1]*np.cos(tool_rotation_angle), 0.0])*np.pi
        # tool_orientation_angle = np.linalg.norm(tool_orientation)
        # tool_orientation_axis = tool_orientation/tool_orientation_angle
        # tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]

        # tilt_rotm = utils.euler2rotm(np.asarray([-np.pi/4,0,0]))
        # tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
        # tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
        # tilted_tool_orientation = tilted_tool_orientation_axis_angle[0]*np.asarray(tilted_tool_orientation_axis_angle[1:4])

        # # Move to box grabbing position
        # # box_grab_position = [0.5,-0.35,-0.12]
        # box_grab_position = [0.207,0.163,0.07]
        # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        # tcp_command = "def process():\n"
        # tcp_command += " set_digital_out(8,False)\n"
        # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2]+0.1,tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc,self.joint_vel)
        # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc,self.joint_vel)
        # tcp_command += " set_digital_out(8,True)\n"
        # tcp_command += "end\n"
        # self.tcp_socket.send(str.encode(tcp_command))
        # self.tcp_socket.close()

        # # Block until robot reaches box grabbing position and gripper fingers have stopped moving
        # state_data = self.get_state()
        # tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        # while True:
        #     state_data = self.get_state()
        #     new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        #     actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
        #     if tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - box_grab_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
        #         break
        #     tool_analog_input2 = new_tool_analog_input2

        # # Move to box release position
        # box_release_position = [0.5,0.08,-0.12]
        # home_position = [0.49,0.11,0.03]
        # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        # tcp_command = "def process():\n"
        # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0],box_release_position[1],box_release_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0],box_release_position[1],box_release_position[2]+0.3,tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.02,self.joint_vel*0.02)
        # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.29)\n" % (box_grab_position[0]-0.05,box_grab_position[1]+0.1,box_grab_position[2]+0.3,tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
        # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0]-0.05,box_grab_position[1]+0.1,box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.5,self.joint_vel*0.5)
        # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0]+0.05,box_grab_position[1],box_grab_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc*0.1,self.joint_vel*0.1)
        # tcp_command += " set_digital_out(8,False)\n"
        # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0],box_grab_position[1],box_grab_position[2]+0.1,tilted_tool_orientation[0],tilted_tool_orientation[1],tilted_tool_orientation[2],self.joint_acc,self.joint_vel)
        # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (home_position[0],home_position[1],home_position[2],tool_orientation[0],tool_orientation[1],tool_orientation[2],self.joint_acc,self.joint_vel)
        # tcp_command += "end\n"
        # self.tcp_socket.send(str.encode(tcp_command))
        # self.tcp_socket.close()

        # # Block until robot reaches home position
        # state_data = self.get_state()
        # tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        # while True:
        #     state_data = self.get_state()
        #     new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        #     actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
        #     if tool_analog_input2 > 3.0 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
        #         break
        #     tool_analog_input2 = new_tool_analog_input2

    def shutdown(self):
        if self.is_sim:
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxFinish(-1)

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


























# JUNK

# command = "movel(p[%f,%f,%f,%f,%f,%f],0.5,0.2,0,0,a=1.2,v=0.25)\n" % (-0.5,-0.2,0.1,2.0171,2.4084,0)

# import socket

# HOST = "192.168.1.100"
# PORT = 30002
# s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# s.connect((HOST,PORT))

# j0 = 0
# j1 = -3.1415/2
# j2 = 3.1415/2
# j3 = -3.1415/2
# j4 = -3.1415/2
# j5 = 0;

# joint_acc = 1.2
# joint_vel = 0.25

# # command = "movej([%f,%f,%f,%f,%f,%f],a=%f,v=%f)\n" % (j0,j1,j2,j3,j4,j5,joint_acc,joint_vel)



# #


# # True closes
# command = "set_digital_out(8,True)\n"

# s.send(str.encode(command))
# data = s.recv(1024)



# s.close()
# print("Received",repr(data))





# print()

# String.Format ("movej([%f,%f,%f,%f,%f, %f], a={6}, v={7})\n", j0, j1, j2, j3, j4, j5, a, v);
