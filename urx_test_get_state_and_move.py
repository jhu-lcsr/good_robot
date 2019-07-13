# https://github.com/SintefManufacturing/python-urx/blob/master/urx/urrobot.py
# Must be run with library from the repo
import urx
import logging
import time
import os
import sys

tcp_host_ip = "10.75.15.91"

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN, stream=sys.stdout)

    rob = urx.Robot(tcp_host_ip)
    #rob = urx.Robot("localhost")

    rob.set_tcp((0, 0, 0, 0, 0, 0))
    rob.set_payload(0.5, (0, 0, 0))
    try:
        delta = 0.05
        v = 0.05
        a = 0.3
        pose = rob.getl()
        print("robot tcp is at: ", pose, '\n')

        print("absolute move in base coordinate ")
        pose[2] += delta
        rob.movel(pose, acc=a, vel=v)
        pose = rob.getl()
        print("robot tcp is at: ", pose, '\n')

        time.sleep(1)

        print("relative move in base coordinate ")
        rob.translate((0, 0, -delta), acc=a, vel=v, relative=True)
        pose = rob.getl()
        print("robot tcp is at: ", pose, '\n')

        time.sleep(10)

        print("relative move back and forth in tool coordinate")
        rob.translate_tool((0, 0, -delta), acc=a, vel=v)
        pose = rob.getl()
        print("robot tcp is at: ", pose, '\n')

        print("relative move back and forth in tool coordinate")
        rob.translate_tool((0, 0, delta), acc=a, vel=v)
        pose = rob.getl()
        print("robot tcp is at: ", pose, '\n')
    except Exception as e:  # RobotError, ex:
        print("Robot could not execute move (emergency stop for example), do something", e)
        rob.close()
        print('closing robot')
        # print('exiting')
        sys.exit()
        print('exiting again')
        os._exit(1)
    finally:
        rob.close()
