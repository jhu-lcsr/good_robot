#! /usr/bin/env python

"""
Python library to control Robotiq Two Finger Gripper connected to UR robot via
Python-URX

Tested using a UR5 Version CB3 and Robotiq 2-Finger Gripper Version 85

SETUP

You must install the driver first and then power on the gripper from the
gripper UI. The driver can be found here:

http://support.robotiq.com/pages/viewpage.action?pageId=5963876

FAQ

Q: Why does this class group all the commands together and run them as a single
program as opposed to running each line seperately (like most of URX)?

A: The gripper is controlled by connecting to the robot's computer (TCP/IP) and
then communicating with the gripper via a socket (127.0.0.1:63352).  The scope
of the socket is at the program level.  It will be automatically closed
whenever a program finishes.  Therefore it's important that we run all commands
as a single program.

DOCUMENTATION

- This code was developed by downloading the gripper package "DCU-1.0.10" from
  http://support.robotiq.com/pages/viewpage.action?pageId=5963876. Or more
  directly from http://support.robotiq.com/download/attachments/5963876/DCU-1.0.10.zip
- The file robotiq_2f_gripper_programs_CB3/rq_script.script was referenced to
  create this class

FUTURE FEATURES

Though I haven't developed it yet, if you look in
robotiq_2f_gripper_programs_CB3/advanced_template_test.script and view function
"rq_get_var" there is an example of how to determine the current state of the
gripper and if it's holding an object.
"""  # noqa

import sys
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper

if __name__ == '__main__':
    rob = urx.Robot("192.168.1.155")
    robotiqgrip = Robotiq_Two_Finger_Gripper(rob)

    # if(len(sys.argv) != 2):
    #     print ("false")
    #     sys.exit()

    # if(sys.argv[1] == "close") :
    #     robotiqgrip.close_gripper()
    # if(sys.argv[1] == "open") :
    robotiqgrip.open_gripper()

    rob.send_program(robotiqgrip.ret_program_to_run())

    rob.close()
    print ("true")
    sys.exit()