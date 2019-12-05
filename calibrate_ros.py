import rospy
import roslib
import numpy as np
import cv2
from robot import Robot
from ros_aruco import ROSArUcoCalibrate
from tqdm import tqdm

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

    def get_rgb_depth_image_and_transform(self):
        color_img, depth_img = self.robot.get_camera_data()
        aruco_tf = self.robot.camera.get_aruco_tf()
        return color_img, depth_img, aruco_tf

    def test(self):
        tool_position=[0.5, -0.3, 0.17]
        tool_orientation = [0, np.pi/2, 0.0] # Real Good Robot

        self.robot.move_to(tool_position, tool_orientation)

        while True:
            color_img, depth_img, aruco_tf = self.get_rgb_depth_image_and_transform()
            cv2.imshow("color.png", color_img)
            cv2.waitKey(1)
            print(aruco_tf.transforms)
    
    def calibrate(self):
        
        workspace_limits = np.asarray([[0.5, 0.75], [-0.3, 0.1], [0.17, 0.3]]) # Real Good Robot
        calib_grid_step = 0.05
        # Checkerboard tracking point offset from the tool in the robot coordinate
        checkerboard_offset_from_tool = [-0.01, 0.0, 0.108]
        # flat and level with fiducial facing up: [0, np.pi/2, 0.0]
        tool_orientations = [[0, np.pi/2, 0.0], [0, 3.0*np.pi/4.0, 0.0], [0, 5.0*np.pi/8.0, 0.0], [0, 5.0*np.pi/8.0, np.pi/8], [np.pi/8.0, 5.0*np.pi/8.0, 0.0]] # Real Good Robot

        # Slow down robot
        self.robot.joint_acc = 1.7
        self.robot.joint_vel = 1.2

        # Construct 3D calibration grid across workspace
        gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], (workspace_limits[0][1] - workspace_limits[0][0])/calib_grid_step)
        gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], (workspace_limits[1][1] - workspace_limits[1][0])/calib_grid_step)
        gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], (workspace_limits[2][1] - workspace_limits[2][0])/calib_grid_step)
        calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
        num_calib_grid_pts = calib_grid_x.shape[0]*calib_grid_x.shape[1]*calib_grid_x.shape[2]
        calib_grid_x.shape = (num_calib_grid_pts,1)
        calib_grid_y.shape = (num_calib_grid_pts,1)
        calib_grid_z.shape = (num_calib_grid_pts,1)
        calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)
        rate = rospy.Rate(0.5) # hz
        robot_poses = []
        marker_poses = []

        for calib_pt_idx in tqdm(range(num_calib_grid_pts)):
            tool_position = calib_grid_pts[calib_pt_idx,:]
            for tool_orientation in tqdm(tool_orientations):
                # print(' pos: ' + str(tool_position) + ' rot: ' + str(tool_orientation))
                self.robot.move_to(tool_position, tool_orientation)
                rate.sleep()
                color_img, depth_img, aruco_tf = self.get_rgb_depth_image_and_transform()
                cv2.imshow("color.png", color_img)
                cv2.waitKey(1)
                if len(aruco_tf.transforms) > 0:
                    # TODO(ahundt) convert tool position and rpy orientation into position + quaternion
                    robot_poses += [[tool_position + tool_orientation]]
                    transform = aruco_tf.transforms[0]
                    marker_poses += [[transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z, 
                                    transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]]
                # TODO(ahundt) get several image transforms at each location and ensure position and orientation noise within some bounds
                # print(aruco_tf.transforms)
        print("robot poses: " + str(robot_poses))
        print("marker poses: " + str(marker_poses))

if __name__ == "__main__":
    calib = Calibrate()
    calib.calibrate()