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

    def calibrate(self):
        tool_position=[0.5, -0.3, 0.17]
        tool_orientation = [0, np.pi/2, 0.0] # Real Good Robot

        self.robot.move_to(tool_position, tool_orientation)

        while True:
            color_img, depth_img, aruco_tf = self.get_rgb_depth_image_and_transform()
            cv2.imshow("color.png", color_img)
            cv2.waitKey(1)
            print(aruco_tf.transforms)

if __name__ == "__main__":
    calib = Calibrate()
    calib.calibrate()