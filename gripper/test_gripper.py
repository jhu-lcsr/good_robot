#! /usr/bin/env python3

'''

Created on Tue Oct 01 2019
Testing the gripper for the Real Good Robot
@author: Hongtao Wu

Using the pyroboticgripper as the gripper driver
'''

from gripper import RobotiqGripper
import time

def main(portname):

    # Initialize the gripper
    RG = RobotiqGripper(portname)

    # Reset and Activate the gripper
    time.sleep(1)
    print('Resetting...')
    RG.reset()
#    RG.printInfo()
    print('Activating...')
    RG.activate()
    RG.printInfo()

#    RG.readAll()
#    print('##################')
#    print([RG.registerDic[key][RG.paramDic[key]] for key in RG.paramDic.keys()])
#    print('##################')
#
#    pos, curr = RG.getPositionCurrent()
#    
#    print('##################')
#    print(f'position: {pos}, current: {curr}')
#    print('##################')
    # Go to a test position
    RG.goTo(20)
    RG.printInfo()
    RG.goTo(230)
    time.sleep(1)

    # # Open the gripper
    # RG.closeGripper()
    # time.sleep(1)

if __name__ == "__main__":
    main('/dev/ttyUSB0')
