#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from robot import Robot
import threading
import os
import utils
from logger import Logger
# TODO(adit98) put this in utils/get from somewhere else?
from generate_sim_stacking_demo import get_and_save_images
import argparse

class HumanControlOfRobot(object):
    """Creates a color and depth opencv window from the robot camera, gets human keyboard/click, and moves the robot.

    Keyboard Controls:
        'c': set self.stop to True indicating it is time to exit the program.
        'a': autonomous mode, sets self.human_control = False, clicks will have no effect (unless move_robot=False).
        'z': human control mode, sets self.human_control = True, clicks will move the robot (unless move_robot=False).
        'h': go home.
        'j': stay in place after pushing, grasping, and placing (self.go_home=False).
        'm': Automatically home when pushing, grasping, and placing (self.go_home=True).
        'g': set self.action = 'grasp', left click in the 'color' image window will do a grasp action.
        'p': set self.action = 'place', left click will do a place action.
        's': set self.action = 'push', left click will slide the gripper across the ground, aka a push action.
        't': set self.action = 'touch', left click will do a touch action (go to a spot and stay there).
        'r': repeat the previous action and click location after applying any settings changes you made to action/angle.
        '1-9': Set the gripper rotation orientation at 45 degree increments, starting at the angle 0. Default is '5'.
        'b': set self.action = box, left click will move the robot to go get the box and dump the objects inside.
        '[': set self.robot.place_task = False, a successful grasp will immediately drop objects in the box.
        ']': set self.robot.place_task = True, a successful grasp will hold on to objects so the robot can place them.
        ' ': print the current robot cartesian position with xyz and axis angle and the current joint angles.
        '-': close gripper
        '=': open gripper
        'k': calibrate with the ros api in calibrate_ros.py

    Member Variables:

        self.stop: if True shut down your program, pressing 'c' on the keyboard sets this variable to True.
    """
    def __init__(self, robot=None, action='touch', human_control=True, mutex=None, move_robot=True,
            logger=None, task_type=None, save_img=False):
        self.stop = False
        self.print_state_count = 0
        self.tool_orientation = [0.0, np.pi, 0.0] # Real Good Robot
        self.human_control = human_control
        self.move_robot = move_robot
        self.action = action
        self.logger = logger
        self.all_action_log = []
        self.successful_action_log = []
        self.heightmap_pairs = []
        self.img_pairs = []
        self.trial = 0
        self.click_count = 0
        self.click_position = None
        self.target_position = None
        # go home automatically during push, grasp place actions
        self.go_home = True
        self.calib = None
        self.task_type = task_type
        self.save_img = save_img

        if robot is None:

            # workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
            # self.tool_orientation = [2.22,-2.22,0]
            # ---------------------------------------------
            # Move robot to home pose
            self.robot = Robot(False, None, None, workspace_limits,
                               tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                               False, None, None)
            robot.open_gripper()
        else:
            self.robot = robot

        if is_sim:
            # reset objects so we have correct progress
            self.reset_sim()

        # Slow down robot
        # robot.joint_acc = 1.4
        # robot.joint_vel = 1.05
        self.grasp_angle = 4.0
        self.grasp_success, self.grasp_color_success = False, False
        self.place_success = False
        if mutex is None:
            self.mutex = threading.Lock()
        # Callback function for clicking on OpenCV window
        self.click_point_pix = ()

        # wait a second for things to initialize
        time.sleep(1)
        self.camera_color_img, self.camera_depth_img = robot.get_camera_data(go_home=False)
        def mouseclick_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                # global camera, robot, self.click_point_pix, action, self.grasp_angle, self.grasp_success, self.grasp_color_success, self.mutex
                self.click_point_pix = (x,y)

                # Get click point in camera coordinates
                click_z = self.camera_depth_img[y][x] * robot.cam_depth_scale * 1000 # unit from m -> mm
                click_x = np.multiply(x-robot.cam_intrinsics[0][2],click_z/robot.cam_intrinsics[0][0])
                click_y = np.multiply(y-robot.cam_intrinsics[1][2],click_z/robot.cam_intrinsics[1][1])
                if click_z == 0:
                    print('Click included invalid camera data, ignoring the command.')
                    return
                click_point = np.asarray([click_x,click_y,click_z]) / 1000  # Convert from unit from mm to m
                click_point.shape = (3,1)

                # Convert camera to robot coordinates
                # camera2robot = np.linalg.inv(robot.cam_pose)
                camera2robot = robot.cam_pose  # The transformation matrix is from meter to meter
                target_position = np.dot(camera2robot[0:3,0:3],click_point) + camera2robot[0:3,3:]

                target_position = target_position[0:3,0]
                heightmap_rotation_angle = self.grasp_angle * np.pi / 4
                # print(target_position, self.tool_orientation)

                if not self.human_control:
                    print('Human Control is disabled, press z for human control mode, a for autonomous mode')
                with self.mutex:
                    self.click_position = target_position.copy()
                    self.target_position, heightmap_rotation_angle = self.execute_action(target_position, heightmap_rotation_angle)

        # Show color and depth frames
        cv2.namedWindow('depth')
        cv2.namedWindow('color')
        cv2.setMouseCallback('color', mouseclick_callback)

        self.print_config()

    def execute_action(self, target_position, heightmap_rotation_angle):
        # log env state (demo/trial num is the poststring for saved heightmaps)
        depth_heightmap, color_heightmap, _, color_img, depth_img = get_and_save_images(self.click_count,
                self.robot, self.logger, self.action, save_image=False)
        self.heightmap_pairs.append((depth_heightmap, color_heightmap))
        self.img_pairs.append((depth_img, color_img))

        self.target_position = target_position
        self.click_count += 1
        def grasp(tp, ra, gh):
            # global self.grasp_success, self.grasp_color_success, self.mutex
            with self.mutex:
                # log action (grasp)
                self.all_action_log.append(tp.tolist() + [ra, utils.ACTION_TO_ID['grasp']])
                self.grasp_success, self.grasp_color_success = robot.grasp(tp, ra, go_home=gh)

        def place(tp, ra, gh):
            # global self.grasp_success, self.mutex
            with self.mutex:
                # log action (place)
                self.all_action_log.append(tp.tolist() + [ra, utils.ACTION_TO_ID['place']])
                self.place_success = self.robot.place(tp, ra, go_home=gh)
                self.grasp_success = False

                if self.place_success:
                    # if we had a successful place, add the last 2 actions (grasp and place) to log
                    self.successful_action_log += self.all_action_log[-2:]

                    # get last two pairs of heightmaps
                    heightmap_pairs = self.heightmap_pairs[-2:]
                    depth_grasp, color_grasp = heightmap_pairs[0]
                    depth_place, color_place = heightmap_pairs[1]

                    if self.logger is not None:
                        # save heightmaps (and images)
                        self.logger.save_heightmaps(self.click_count, color_grasp,
                                depth_grasp, 'grasp', poststring=self.trial)
                        self.logger.save_heightmaps(self.click_count, color_place,
                                depth_place, 'place', poststring=self.trial)
                        if self.save_img:
                            img_pairs = self.img_pairs[-2:]
                            depth_grasp_img, color_grasp_img = img_pairs[0]
                            depth_place_img, color_place_img = img_pairs[1]
                            self.logger.save_images(self.click_count, color_grasp_img,
                                    depth_grasp_img, 'grasp')
                            self.logger.save_images(self.click_count, color_place_img,
                                    depth_place_img, 'place')


        if self.action == 'touch':
            # Move the gripper up a bit to protect the gripper (Real Good Robot)
            def move_to(tp, ra):
                # global self.mutex
                tp = tp.copy()
                # move to a spot just above the clicked spot to avoid collision
                tp[-1] += 0.04
                with self.mutex:
                    # self.robot.move_to(target_position, self.tool_orientation)
                    self.robot.move_to(tp, heightmap_rotation_angle=ra)
            if self.move_robot:
                t = threading.Thread(target=move_to, args=(target_position, heightmap_rotation_angle))
                t.start()

        elif self.action == 'grasp':
            # make sure we should be grasping
            if not self.robot.place_task or (self.robot.place_task and not self.grasp_success):
                if self.move_robot:
                    t = threading.Thread(target=grasp, args=(target_position, heightmap_rotation_angle, self.go_home))
                    t.start()

            # above check failed, need to place
            else:
                # adjust z height
                target_position[-1] += 0.01

                if self.move_robot:
                    self.action = 'place'
                    t = threading.Thread(target=place, args=(target_position, heightmap_rotation_angle, self.go_home))
                    t.start()

        elif self.action == 'box':
            t = threading.Thread(target=lambda: self.robot.restart_real())
            t.start()
        elif self.action == 'push':
            target_position[-1] += 0.01
            t = threading.Thread(target=lambda: self.robot.push(target_position, heightmap_rotation_angle, go_home=self.go_home))
            t.start()
        elif self.action == 'place':
            # check if we should be grasping even though place was specified
            if not self.grasp_success:
                t = threading.Thread(target=grasp, args=(target_position, heightmap_rotation_angle, self.go_home))
                t.start()
            else:
                # adjust z height
                target_position[-1] += 0.01
                t = threading.Thread(target=place, args=(target_position, heightmap_rotation_angle, self.go_home))
                t.start()

        print(str(self.click_count) + ': action: ' + str(self.action) + ' pos: ' + str(target_position) + ' rot: ' + str(heightmap_rotation_angle))

        return target_position, heightmap_rotation_angle

    def print_config(self):
        # global robot
        state_str = 'Current action: ' + str(self.action) + '. '
        state_str += 'Grasp, HOLD, PLACE object task, ' if self.robot.place_task else 'Grasp then drop in box task, '
        state_str += 'robot WILL go home after push/grasp/place' if self.go_home else 'robot will NOT go home after push/grasp/place'
        if self.task_type == 'vertical_square':
            print(self.robot.vertical_square_partial_success(np.ones(4), check_z_height=False, stack_dist_thresh=0.04))
            print(state_str)
        elif self.task_type == 'row':
            print(self.robot.check_row(np.ones(4), check_z_height=False))
            print(state_str)
        else:
            print(state_str)

    def run_one(self, camera_color_img=None, camera_depth_img=None):
        if camera_color_img is None:
            shape = [0, 0, 0, 0]
            # get the camera data, but make sure all the images are valid first
            while not all(shape):
                self.camera_color_img, self.camera_depth_img = self.robot.get_camera_data(go_home=False)
                shape = self.camera_color_img.shape + self.camera_depth_img.shape
        else:
            self.camera_color_img = camera_color_img
            self.camera_depth_img = camera_depth_img

        if len(self.click_point_pix) != 0:
            self.camera_color_img = cv2.circle(self.camera_color_img, self.click_point_pix, 7, (0,0,255), 2)
        self.camera_color_img = cv2.cvtColor(self.camera_color_img, cv2.COLOR_RGB2BGR)
        cv2.imshow('color', self.camera_color_img)
        cv2.imshow('depth', self.camera_depth_img)

        key = cv2.waitKey(1)
        # Configure the system
        # Numbers 1-9 are orientations of the gripper
        # t is touch mode, where the robot will go to the clicked spot

        if key == ord('1'):
            self.grasp_angle = 0.0
        elif key == ord('2'):
            self.grasp_angle = 1.0
        elif key == ord('3'):
            self.grasp_angle = 2.0
        elif key == ord('4'):
            self.grasp_angle = 3.0
        elif key == ord('5'):
            self.grasp_angle = 4.0
        elif key == ord('6'):
            self.grasp_angle = 5.0
        elif key == ord('7'):
            self.grasp_angle = 6.0
        elif key == ord('8'):
            self.grasp_angle = 7.0
        elif key == ord('9'):
            self.grasp_angle = 8.0
        elif key == ord('t'):
            self.action = 'touch'
            self.print_config()
        elif key == ord('g'):
            self.action = 'grasp'
            self.print_config()
        elif key == ord('s'):
            self.action = 'push'
            self.print_config()
        elif key == ord('p'):
            self.action = 'place'
            self.print_config()
        elif key == ord('b'):
            self.action = 'box'
            self.print_config()
        elif key == ord('r'):
            heightmap_rotation_angle = self.grasp_angle * np.pi / 4
            with self.mutex:
                self.target_position, heightmap_rotation_angle = self.execute_action(self.click_position.copy(), heightmap_rotation_angle)
        elif key == ord(']'):
            with self.mutex:
                # Mode for stacking blocks
                self.robot.place_task = True
                self.print_config()
        elif key == ord('['):
            with self.mutex:
                # Mode for grasping to hold and then place
                self.robot.place_task = False
                self.print_config()
        elif key == ord(' '):
            with self.mutex:
                # print the robot state
                self.print_state_count += 1
                state_data = self.robot.get_state()
                actual_tool_pose = self.robot.parse_tcp_state_data(state_data, 'cartesian_info')
                robot_state = 'UR5 axis/angle cart_pose format: ' + str(actual_tool_pose)
                actual_tool_pose = utils.axis_angle_and_translation_to_rigid_transformation(actual_tool_pose[:3], actual_tool_pose[3:])
                joint_position = self.robot.parse_tcp_state_data(state_data, 'joint_data')
                robot_state += ' joint pos: ' + str(joint_position) + ' homogeneous cart_pose: ' + str(actual_tool_pose)
                print(str(self.print_state_count) + ' ' + robot_state)
        elif key == ord('f'):
            if self.logger is not None:
                # finish trial, write actions (only the successful ones), and move to next trial
                self.logger.write_to_log('executed-actions-' + str(self.trial),
                        self.successful_action_log)
                self.logger.write_to_log('all-actions-' + str(self.trial), self.all_action_log)
            self.trial += 1

            # clear logs
            self.all_action_log = []
            self.successful_action_log = []

            # NOTE(adit98) figure out how to show next img before action selection
            # finish trial and move to next trial
            # keep repositioning objects until we start from 1st step
            if is_sim:
                self.reset_sim()

        elif key == ord('c'):
            if self.logger is not None:
                # we stopped the program, write actions (only the successful ones)
                self.logger.write_to_log('executed-actions-' + str(self.trial),
                        self.successful_action_log)
                self.logger.write_to_log('all-actions-' + str(self.trial), self.all_action_log)

                # write current heightmap
                depth_heightmap, color_heightmap, _, color_img, depth_img = get_and_save_images(self.click_count,
                        self.robot, self.logger, self.action, save_image=False)
                self.logger.save_heightmaps(self.click_count, color_heightmap,
                        depth_heightmap, 'end', poststring=self.trial)
                if self.save_img:
                    self.logger.save_images(self.click_count, color_grasp_img,
                            depth_grasp_img, 'end')

            self.stop = True
        elif key == ord('h'):
            with self.mutex:
                t = threading.Thread(target=lambda: self.robot.go_home())
                t.start()
        elif key == ord('-'):
            with self.mutex:
                t = threading.Thread(target=lambda: print('fully closed: ' + str(self.robot.close_gripper()) + \
                        ' obj detected: ' + str(self.robot.gripper.object_detected())))
                t.start()
        elif key == ord('='):
            with self.mutex:
                t = threading.Thread(target=lambda: self.robot.open_gripper())
                t.start()
        elif key == ord('m'):
            self.go_home = True
            self.print_config()
        elif key == ord('j'):
            self.go_home = False
            self.print_config()
        elif key == ord('z'):
            self.human_control = True
        elif key == ord('a'):
            self.human_control = False
        elif key == ord('k'):
            from calibrate_ros import Calibrate
            robot.camera.subscribe_aruco_tf()
            robot.go_home()
            calib = Calibrate(robot=self.robot)
            # calib.test()
            calib.calibrate()
            # def calibration():
            #     from calibrate_ros import Calibrate
            #     robot.camera.subscribe_aruco_tf()
            #     robot.go_home()
            #     calib = Calibrate(robot=self.robot)
            #     # calib.test()
            #     calib.calibrate()

            # with self.mutex:
            #     t = threading.Thread(target=calibration)
            #     t.start()

    def run(self):
        """ Blocking call that repeatedly calls run_one()
        """
        while not hcr.stop:
            hcr.run_one()

    def __del__(self):
        cv2.destroyAllWindows()

    def reset_sim(self):
        while True:
            self.robot.reposition_objects()
            if self.task_type == 'vertical_square':
                _, progress = self.robot.vertical_square_partial_success(np.ones(4), check_z_height=False, stack_dist_thresh=0.04)
            elif self.task_type == 'row':
                _, progress = self.robot.check_row(np.ones(4), check_z_height=False)
            elif self.task_type == 'stack':
                _, progress = self.robot.check_stack(np.ones(4))
            else:
                progress = 1

            if progress == 1:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_type', default='stack', type=str)
    parser.add_argument('-s', '--save_img', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    args = parser.parse_args()

    # User options (change me)
    # --------------- Setup options ---------------
    tcp_host_ip = '192.168.1.155' # IP and port to robot arm as TCP client (UR5)
    tcp_port = 30002
    rtc_host_ip = '192.168.1.155' # IP and port to robot arm as real-time client (UR5)
    rtc_port = 30003
    # action = 'touch'
    action = 'grasp'

    if action == 'touch':
        # workspace_limits = np.asarray([[0.5, 0.75], [-0.3, 0.1], [0.17, 0.3]]) # Real Good Robot
        workspace_limits = None
    elif action == 'grasp':
        workspace_limits = None
    else:
        raise NotImplementedError
    is_sim = True
    if is_sim:
        tcp_port = 19997
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)

    heightmap_resolution = 0.002 # Meters per pixel of heightmap
    calibrate = False
    # Move robot to home pose
    # TODO(adit98) add cmd line args to select goal, task, etc.
    robot = Robot(is_sim, os.path.abspath('objects/blocks'), 4, workspace_limits,
                tcp_host_ip, tcp_port, rtc_host_ip, rtc_port, False, None, None,
                place=True, calibrate=calibrate, unstack=False, task_type=args.task_type)

    if args.save:
        # initialize logger
        logger = Logger(continue_logging=False, logging_directory='demos')
        logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
        logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters
    else:
        logger = None

    hcr = HumanControlOfRobot(robot, action=action, logger=logger, task_type=args.task_type, save_img=args.save_img)
    hcr.run()
