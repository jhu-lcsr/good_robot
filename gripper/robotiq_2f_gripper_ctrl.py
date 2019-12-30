#!/usr/bin/env python3

"""
Control the Robotiq 2f-85 gripper based on robotiq_2f_gripper_ctrl.py in: https://github.com/ros-industrial/robotiq
@author: Hongtao Wu
Oct 12, 2019
"""
import numpy as np
from .robotiq_2f_gripper_control_msg import outputMsg
from .robotiq_2f_gripper_control_msg import inputMsg
from .baseRobotiq2FGripper import robotiqbaseRobotiq2FGripper
from .comModbusTcp import communication
import time


class RobotiqCGripper(object):
    def __init__(self, address):
        self.cur_status = None
        #Gripper is a 2F with a TCP connection
        self.gripper = robotiqbaseRobotiq2FGripper()
        self.gripper.client = communication()

        #We connect to the address received as an argument
        self.gripper.client.connectToDevice(address)

    def update(self, outputmsg):
        self.gripper.refreshCommand(outputmsg)
        #Get and publish the Gripper status
        self.cur_status = self.gripper.getStatus()     

        #Wait a little
        time.sleep(0.05)

        #Send the most recent command
        self.gripper.sendCommand()
        
        #Wait a little
        time.sleep(0.05)
    
    def get_cur_status(self):
        self.cur_status = self.gripper.getStatus()

    def wait_for_connection(self):
        # rospy.sleep(0.1)
        # r = rospy.Rate(30)
        # start_time = rospy.get_time()
        # while not rospy.is_shutdown():
        #     if (timeout >= 0. and rospy.get_time() - start_time > timeout):
        #         return False
        #     if self.cur_status is not None:
        #         return True
        #     r.sleep()
        # return False
        time.sleep(0.1)
        self.get_cur_status()
        if self.cur_status:
            print("Successfully connected to the gripper!")
            return True
        else:
            print("Failed to connect to the gripper!")
            return False

    def is_ready(self):
        self.get_cur_status()
        return self.cur_status.gSTA == 3 and self.cur_status.gACT == 1

    def is_reset(self):
        self.get_cur_status()
        return self.cur_status.gSTA == 0 or self.cur_status.gACT == 0

    def is_moving(self):
        self.get_cur_status()   
        return self.cur_status.gGTO == 1 and self.cur_status.gOBJ == 0

    def is_stopped(self):
        self.get_cur_status()
        return self.cur_status.gOBJ != 0

    def object_detected(self):
        self.get_cur_status()
        return self.cur_status.gOBJ == 1 or self.cur_status.gOBJ == 2

    def get_fault_status(self):
        self.get_cur_status()
        return self.cur_status.gFLT

    def get_pos(self):
        self.get_cur_status()
        po = self.cur_status.gPO
        return np.clip(0.087/(13.-230.)*(po-230.), 0, 0.087)

    def get_req_pos(self):
        self.get_cur_status()
        pr = self.cur_status.gPR
        return np.clip(0.087/(13.-230.)*(pr-230.), 0, 0.087)

    def is_closed(self):
        self.get_cur_status()
        return self.cur_status.gPO >= 230

    def is_opened(self):
        self.get_cur_status()
        return self.cur_status.gPO <= 13

    # in mA
    def get_current(self):
        self.get_cur_status()
        return self.cur_status.gCU * 0.1

    # if timeout is negative, wait forever
    def wait_until_stopped(self, timeout):
        #  r = rospy.Rate(30)
        # start_time = rospy.get_time()
        # while not rospy.is_shutdown():
        #     if (timeout >= 0. and rospy.get_time() - start_time > timeout) or self.is_reset():
        #         return False
        #     if self.is_stopped():
        #         return True
        #     r.sleep()
        # return False
        if timeout < 0:
            raise ValueError("Please set the timeout!")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_stopped():
                return True
        return False

    def wait_until_moving(self, timeout=-1):
        # r = rospy.Rate(30)
        # start_time = rospy.get_time()
        # while not rospy.is_shutdown():
        #     if (timeout >= 0. and rospy.get_time() - start_time > timeout) or self.is_reset():
        #         return False
        #     if not self.is_stopped():
        #         return True
        #     r.sleep()
        # return False
        raise NotImplementedError("This method has not been implemented!")

    def reset(self):
        cmd = outputMsg()
        cmd.rACT = 0
        # self.cmd_pub.publish(cmd)
        self.update(cmd)

    def activate(self, timeout=30):
        cmd = outputMsg()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rPR = 0
        cmd.rSP = 255
        cmd.rFR = 150
        # self.cmd_pub.publish(cmd)
        self.update(cmd)
        # r = rospy.Rate(30)
        # start_time = rospy.get_time()
        # while not rospy.is_shutdown():
        #     if timeout >= 0. and rospy.get_time() - start_time > timeout:
        #         return False
        #     if self.is_ready():
        #         return True
        #     r.sleep()
        # return False
        startTime = time.time()
        while time.time() - startTime < timeout:
            if self.is_ready():
                print("The gripper is activated!")
                return True
        return False

    def auto_release(self):
        cmd = outputMsg()
        cmd.rACT = 1
        cmd.rATR = 1
        # self.cmd_pub.publish(cmd)

    ##
    # Goto position with desired force and velocity
    # @param pos Gripper width in meters. [0, 0.087]
    # @param vel Gripper speed in m/s. [0.013, 0.100]
    # @param force Gripper force in N. [30, 100] (not precise)
    def goto(self, pos, vel, force, timeout, block=False):
        cmd = outputMsg()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rPR = int(np.clip((13.-230.)/0.087 * pos + 230., 0, 255))
        cmd.rSP = int(np.clip(255./(0.1-0.013) * (vel-0.013), 0, 255))
        cmd.rFR = int(np.clip(255./(100.-30.) * (force-30.), 0, 255))
        # self.cmd_pub.publish(cmd)
        self.update(cmd)
        if timeout < 0:
            raise ValueError("Please set timeout!")
        # rospy.sleep(0.1)
        # if block:
        #     if not self.wait_until_moving(timeout):
        #         return False
        #     return self.wait_until_stopped(timeout)
        # return True
        result = self.wait_until_stopped(timeout)
        print("Gripper finished moving!")
        return result


    def stop(self, block=False, timeout=-1):
        cmd = outputMsg()
        cmd.rACT = 1
        cmd.rGTO = 0
        # self.cmd_pub.publish(cmd)
        # rospy.sleep(0.1)
        # if block:
        #     return self.wait_until_stopped(timeout)
        # return True

    def open(self, vel=0.1, force=100, block=False, timeout=10):
        if self.is_opened():
            return True
        return self.goto(1.0, vel, force, block=block, timeout=timeout)

    def close(self, vel=0.1, force=100, block=False, timeout=10):
        if self.is_closed():
            print("The gripper is already closed!")
            return True
        return self.goto(-1.0, vel, force, timeout=timeout, block=block)

def main():
    # rospy.init_node("robotiq_2f_gripper_ctrl_test")
    gripper = RobotiqCGripper('192.168.1.11')
    gripper.wait_for_connection()
    # if gripper.is_reset():
    gripper.reset()
    gripper.activate()

    print('Start Sleeping!')
    time.sleep(2)
    print('Close!')
    
    gripper.close(block=True)

    print('Start Sleeping!')
    time.sleep(2)
    print('Open!')

    gripper.open(block=True)
    # while not rospy.is_shutdown():
    #     print gripper.open(block=False)
    #     rospy.sleep(0.11)
    #     gripper.stop()
    #     print gripper.close(block=False)
    #     rospy.sleep(0.1)
    #     gripper.stop()

if __name__ == '__main__':
    main()