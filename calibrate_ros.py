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

    def __init__(self, tcp_host_ip='192.168.1.155', tcp_port=30002, rtc_host_ip='192.168.1.155', rtc_port = 30003):
        self.workspace_limits = np.asarray([[0.5, 0.75], [-0.3, 0.1], [0.17, 0.3]]) # Real Good Robot
        self.calib_grid_step = 0.05
        self.robot = Robot(False, None, None, self.workspace_limits,
              tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
              False, None, None, calibrate=True)
        
        print('Robot active, open the gripper')
    
        self.robot.open_gripper()
        print('Gripper opened!')

        self.robot.joint_acc = 1.7
        self.robot.joint_vel = 1.2

        print('MOVING THE ROBOT to home position...')
        self.robot.go_home()

        self.calibration_file_folder = '/home/costar/src/real_good_robot/calibration'
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
    
    def collect_data(self, calib_grid_step=0.1, workspace_limits=None):
        """
        # Arguments

            calib_grid_step: Meters per cartesian data collection point, reasonable values are 0.1 and 0.05.
        """
        if workspace_limits is None:
            workspace_limits = np.asarray([[0.5, 0.75], [-0.3, 0.1], [0.17, 0.3]]) # Real Good Robot
        # calib_grid_step = 0.05
        calib_grid_step = 0.1
        # Checkerboard tracking point offset from the tool in the robot coordinate
        # checkerboard_offset_from_tool = [-0.01, 0.0, 0.108]
        # flat and level with fiducial facing up: [0, np.pi/2, 0.0]
        tool_orientations = [[0, np.pi/2, 0.0], [0, 3.0*np.pi/4.0, 0.0], [0, 5.0*np.pi/8.0, 0.0], [0, 5.0*np.pi/8.0, np.pi/8], [np.pi/8.0, 5.0*np.pi/8.0, 0.0]] # Real Good Robot

        # Slow down robot
        self.robot.joint_acc = 1.7
        self.robot.joint_vel = 1.2

        # Construct 3D calibration grid across workspace
        num_calib_grid_pts, calib_grid_pts = utils.calib_grid_cartesian(workspace_limits, calib_grid_step)
        rate = rospy.Rate(0.5) # hz
        robot_poses = []
        marker_poses = []

        print("Start Calibrating...")

        for calib_pt_idx in tqdm(range(num_calib_grid_pts)):
            tool_position = calib_grid_pts[calib_pt_idx,:]
            for tool_orientation_idx, tool_orientation in enumerate(tqdm(tool_orientations)):
                tool_orientation_idx += 1
                self.robot.move_to(tool_position, tool_orientation)
                rate.sleep()

                color_img, depth_img, aruco_tf, aruco_img = self.get_rgb_depth_image_and_transform()

                # cv2.imshow("color.png", aruco_img)
                # cv2.waitKey(1)
                img_prefix = str(calib_pt_idx) + '_' + str(tool_orientation_idx)
                aruco_img_file = os.path.join(self.calibration_file_folder, 'arucoimg_' + img_prefix + '.png')
                cv2.imwrite(aruco_img_file, aruco_img)
                rgb_img_file = os.path.join(self.calibration_file_folder, 'rgb_img_' + img_prefix + '.png')
                cv2.imwrite(rgb_img_file, color_img)
                depth_img_file = os.path.join(self.calibration_file_folder, 'depth_img_' + img_prefix + '.png')
                cv2.imwrite(depth_img_file, aruco_img)

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
                
        # print("robot poses: " + str(robot_poses))
        # print("marker poses: " + str(marker_poses))
        return robot_poses, marker_poses

    def save_transforms_to_file(self, calib_pt_idx, tool_orientation_idx, aruco_tf, tool_transformation, marker_transformation):
        
        robot_pose_file = os.path.join(self.calibration_file_folder, str(calib_pt_idx) + '_' + str(tool_orientation_idx) + '_robotpose.txt')
        marker_pose_file = os.path.join(self.calibration_file_folder, str(calib_pt_idx) + '_' + str(tool_orientation_idx) + '_markerpose.txt')
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


    # TODO (Hongtao): After making sure that the tag transformation is correct, make sure that the function is correct
    def solve_axxb_horn(self):
        # Load robot pose and marker pose
        for f in os.listdir(self.calibration_file_folder):
            if 'robotpose.txt' in f:
                robot_pose_file = f
                marker_pose_file = f[:-13] + 'markerpose.txt'
                
                # tool pose in robot base frame
                with open(os.path.join(self.calibration_file_folder, robot_pose_file), 'r') as file_robot:
                    robotpose_str = file_robot.readline().split(' ')
                    robotpose = [float (x) for x in robotpose_str if x is not '']
                    assert len(robotpose) == 16
                    robotpose = np.reshape(np.array(robotpose), (4, 4))
                self.robot_poses.append(robotpose)
                
                # marker pose in camera frame
                with open(os.path.join(self.calibration_file_folder, marker_pose_file), 'r') as file_marker:
                    markerpose_str = file_marker.readline().split(' ')
                    markerpose = [float(x) for x in markerpose_str if x is not '']
                    assert len(markerpose) == 16
                    markerpose = np.reshape(np.array(markerpose), (4, 4))

                import ipdb; ipdb.set_trace()
                self.marker_poses.append(markerpose)
        
        # AX=XB calibration: marker pose in tool frame
        marker2tool = utils.axxb(self.robot_poses, self.marker_poses)
        
        print("Camera in robot base:")
        for i in range(10):
            cam2base = np.matmul(np.matmul(self.robot_poses[i], marker2tool), np.linalg.inv(self.marker_poses[i]))
            print(cam2base)
    
    def calibrate(self):

        # TODO(ahundt) split train and validation sets, add loop for RANSAC consensus
        robot_poses, marker_poses = calib.collect_data()
        X = SolverAXXB.LeastSquareAXXB(robot_poses, marker_poses)
        # error = SolverAXXB.Validation(X, ValidSet)
        np.savetxt(os.path.join(self.calibration_file_folder, 'cam2base.txt'), X)
        print('Final Transform: \n' + str(X))
        return X

    


if __name__ == "__main__":
    calib = Calibrate()
    calib.calibrate()