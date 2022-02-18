
import time
try:
    from gripper.robotiq_2f_gripper_ctrl import RobotiqCGripper
except ImportError:
    print('Real robotiq gripper control is not available. '
          'Ensure pymodbus is installed:\n'
          '    pip3 install --user --upgrade pymodbus\n')
    RobotiqCGripper = None

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
    # for i in range(6):
    #     time.sleep(10)
    #     gripper.get_cur_status()
    time.sleep(60)
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