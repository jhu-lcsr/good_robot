"""
Using aruco_detect ROS package to calibrate eye-on-base camera calibration.
@author: Hongtao Wu, Andrew Hundt
Dec 05, 2019
"""

import rospy
import roslib
import numpy as np
import cv2
from robot import Robot
from ros_aruco import ROSArUcoCalibrate
from tqdm import tqdm
import time
import os
import utils
import SolverAXXB

def ros_transform_to_numpy_transform(transform):
    # Marker quaternion
    marker_quaternion = np.array([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
    # Marker rotation matrix
    marker_orientation_rotm = utils.quat2rotm(marker_quaternion)
    # Marker transformation
    marker_position = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
    # Marker rigid body transformation
    marker_transformation = np.zeros((4, 4))
    marker_transformation[:3, :3] = marker_orientation_rotm
    marker_transformation[:3, 3] = marker_position
    marker_transformation[3, 3] = 1
    return marker_transformation

class Calibrate:

    def __init__(
            self, tcp_host_ip='192.168.1.155', tcp_port=30002, rtc_host_ip='192.168.1.155', rtc_port = 30003, 
            save_dir=None, workspace_limits=None, calib_grid_step=0.06):
        if workspace_limits is None:
            self.workspace_limits = np.asarray([[0.5, 0.75], [-0.3, 0.1], [0.17, 0.3]]) # Real Good Robot
        self.calib_grid_step = calib_grid_step

        # we only activate the robot when we are actually collect data
        self.robot = None
        self.tcp_host_ip = tcp_host_ip
        self.tcp_port = tcp_port
        self.rtc_host_ip = rtc_host_ip
        self.rtc_port = rtc_port

        if save_dir is None:
            # TODO(ahundt) make this path something reasonable, and create the directory if it doesn't exist
            self.save_dir = '/home/costar/src/real_good_robot/calibration'
        else:
            self.save_dir = save_dir

        self.robot_poses = []
        self.marker_poses = []

    def get_rgb_depth_image_and_transform(self):
        color_img, depth_img = self.robot.get_camera_data()
        aruco_tf, aruco_img = self.robot.camera.get_aruco_tf()
        return color_img, depth_img, aruco_tf, aruco_img

    def test(self):
        tool_position=[0.5, -0.3, 0.17]
        tool_orientation = [0, np.pi/2, 0.0] # Real Good Robot

        self.robot.move_to(tool_position, tool_orientation)

        while True:
            color_img, depth_img, aruco_tf = self.get_rgb_depth_image_and_transform()
            cv2.imshow("color.png", color_img)
            cv2.waitKey(1)
            print(aruco_tf.transforms)
    
    def activate_robot(self):
        if self.robot is None:
            print('Activating the robot, prepare for it to move!')
            self.robot = Robot(False, None, None, self.workspace_limits,
                self.tcp_host_ip, self.tcp_port, self.rtc_host_ip, self.rtc_port,
                False, None, None, calibrate=True)
            
            print('Robot active, open the gripper')
        
            self.robot.open_gripper()
            print('Gripper opened!')

            self.robot.joint_acc = 1.7
            self.robot.joint_vel = 1.2

            print('MOVING THE ROBOT to home position...')
            self.robot.go_home()

    
    def collect_data(self, calib_grid_step=None, workspace_limits=None, rate_hz=0.5):
        """
        # Arguments

            calib_grid_step: Meters per cartesian data collection point, reasonable values are 0.06 (default), and 0.05.
                             Smaller step sizes collect more data, but take exponentially longer to run.
            rate_hz: The rate at which each data point is collected. You may want this to be smaller if time is needed to settle in place.
        """

        # Test the destination directory.
        if ( False == os.path.isdir( self.save_dir ) ):
            print("The destination directory (%s) does not exist. Creating the directory." % self.save_dir)
            os.mkdir( self.save_dir )
        if calib_grid_step is None:
            calib_grid_step = self.calib_grid_step
        if workspace_limits is None:
            workspace_limits = self.workspace_limits

        if self.robot is None:
            self.activate_robot()
        
        rate = rospy.Rate(rate_hz) # hz
        rate.sleep()
        # Checkerboard tracking point offset from the tool in the robot coordinate
        # checkerboard_offset_from_tool = [-0.01, 0.0, 0.108]
        # flat and level with fiducial facing up: [0, np.pi/2, 0.0]
        tool_orientations = [[0, np.pi/2, 0.0], [0, 3.0*np.pi/4.0, 0.0], [0, 5.0*np.pi/8.0, 0.0], [0, 5.0*np.pi/8.0, np.pi/8], [np.pi/8.0, 5.0*np.pi/8.0, 0.0]] # Real Good Robot

        # Slow down robot
        self.robot.joint_acc = 1.7
        self.robot.joint_vel = 1.2

        # Construct 3D calibration grid across workspace
        num_calib_grid_pts, calib_grid_pts = utils.calib_grid_cartesian(workspace_limits, calib_grid_step)
        robot_poses = []
        marker_poses = []

        print("Starting Calibration and saving data in: " + str(self.save_dir))
        with tqdm(total=num_calib_grid_pts * len(tool_orientations)) as pbar:
            for calib_pt_idx in range(num_calib_grid_pts):
                tool_position = calib_grid_pts[calib_pt_idx,:]
                for tool_orientation_idx, tool_orientation in enumerate(tool_orientations):
                    tool_orientation_idx += 1
                    self.robot.move_to(tool_position, tool_orientation)
                    rate.sleep()

                    color_img, depth_img, aruco_tf, aruco_img = self.get_rgb_depth_image_and_transform()

                    # cv2.imshow("color.png", aruco_img)
                    # cv2.waitKey(1)
                    img_prefix = str(calib_pt_idx) + '_' + str(tool_orientation_idx)
                    aruco_img_file = os.path.join(self.save_dir, 'arucoimg_' + img_prefix + '.png')
                    cv2.imwrite(aruco_img_file, aruco_img)
                    rgb_img_file = os.path.join(self.save_dir, 'rgb_img_' + img_prefix + '.png')
                    cv2.imwrite(rgb_img_file, color_img)
                    depth_img_file = os.path.join(self.save_dir, 'depth_img_' + img_prefix + '.png')
                    cv2.imwrite(depth_img_file, depth_img)

                    if len(aruco_tf.transforms) > 0:
                        # TODO(ahundt) we assume only one marker is visible, at least check that the id matches the whole time.
                        transform = aruco_tf.transforms[0]
                        # TODO (Hongtao): make sure that the transformations of robot and tag are correct
                        tool_transformation = utils.axis_angle_and_translation_to_rigid_transformation(tool_position, tool_orientation)
                        robot_poses += [tool_transformation]
                        marker_transformation = ros_transform_to_numpy_transform(transform)
                        marker_poses += [marker_transformation]
                        # TODO(ahundt) need cleaner separation of concerns, only save file once with np.savetxt, pandas, or some other one liner
                        self.save_transforms_to_file(calib_pt_idx, tool_orientation_idx, aruco_tf, tool_transformation, marker_transformation)
                    pbar.update()
                
        # print("robot poses: " + str(robot_poses))
        # print("marker poses: " + str(marker_poses))
        return robot_poses, marker_poses

    def save_transforms_to_file(self, calib_pt_idx, tool_orientation_idx, aruco_tf, tool_transformation, marker_transformation):
        
        robot_pose_file = os.path.join(self.save_dir, str(calib_pt_idx) + '_' + str(tool_orientation_idx) + '_robotpose.txt')
        marker_pose_file = os.path.join(self.save_dir, str(calib_pt_idx) + '_' + str(tool_orientation_idx) + '_markerpose.txt')
        if len(aruco_tf.transforms) > 0:
            # Tool pose in robot base frame
            with open(robot_pose_file, 'w') as file1:

                # print "Robot tool transformation"
                # print tool_transformation

                for l in np.reshape(tool_transformation, (16, )).tolist():
                    file1.writelines(str(l) + ' ')

            # Marker pose in camera frame
            with open(marker_pose_file, 'w') as file2:
                # Marker quaternion
                # marker_transformation = utils.make_rigid_transformation(marker_position, marker_quaternion)

                # print "Marker transformation"
                # print marker_transformation

                for l in np.reshape(marker_transformation, (16, )).tolist():
                    file2.writelines(str(l) + ' ')

    # TODO(Hongtao): After making sure that the tag transformation is correct, make sure that the function is correct
    def calibrate(self, robot_poses=None, marker_poses=None, load_dir=None):
        """Perform calibration

        robot_poses: a list of 4x4 transforms from the robot base to the tool tip
        marker_poses: a list of 4x4 transforms from the camera to the AR tag marger
        """
        if load_dir is not None:
            # Load robot pose and marker pose
            robot_poses, marker_poses = self.load_marker_poses(load_dir)
        elif robot_poses is None and marker_poses is None:
            # collect the pose data
            robot_poses, marker_poses = calib.collect_data()

        
        # TODO(ahundt) split train and validation sets, add loop for RANSAC consensus
        # AX=XB calibration: marker pose in tool frame
        # marker2tool = utils.axxb(self.robot_poses, self.marker_poses)
        marker2tool = SolverAXXB.LeastSquareAXXB(robot_poses, marker_poses)
        # error = SolverAXXB.Validation(X, ValidSet)
        np.savetxt(os.path.join(self.save_dir, 'marker2tool.txt'), marker2tool)
        
        for i in range(len(robot_poses)):
            print("Camera in robot base example" + str(i) + ":")
            cam2base = np.matmul(np.matmul(self.robot_poses[i], marker2tool), np.linalg.inv(self.marker_poses[i]))
            print(cam2base)
            np.savetxt(os.path.join(self.save_dir, 'cam2base_' + str(i) + '.txt'), marker2tool)
        # TODO(ahundt) Make sure this isn't actuall Marker to Tool
        print('Final Marker to Tool Transform : \n' + str(marker2tool))

        return marker2tool

    def load_marker_poses(self, load_dir):
        """ Load robot pose and marker pose from a save directory
        """
        robot_poses = []
        marker_poses = []
        for f in os.listdir(self.save_dir):
            if 'robotpose.txt' in f:
                robot_pose_file = f
                # TODO(hongtao) this next line probably breaks when the number of digits in samples changes (9, 99, 999)
                marker_pose_file = f[:-13] + 'markerpose.txt'

                # tool pose in robot base frame
                with open(os.path.join(self.save_dir, robot_pose_file), 'r') as file_robot:
                    robotpose_str = file_robot.readline().split(' ')
                    robotpose = [float (x) for x in robotpose_str if x is not '']
                    assert len(robotpose) == 16
                    robotpose = np.reshape(np.array(robotpose), (4, 4))
                robot_poses.append(robotpose)

                # marker pose in camera frame
                with open(os.path.join(self.save_dir, marker_pose_file), 'r') as file_marker:
                    markerpose_str = file_marker.readline().split(' ')
                    markerpose = [float(x) for x in markerpose_str if x is not '']
                    assert len(markerpose) == 16
                    markerpose = np.reshape(np.array(markerpose), (4, 4))

                # import ipdb; ipdb.set_trace()
                marker_poses.append(markerpose)
        return robot_poses, marker_poses 

    


if __name__ == "__main__":
    calib = Calibrate()
    calib.calibrate()