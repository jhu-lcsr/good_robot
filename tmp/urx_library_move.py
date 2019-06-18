# https://github.com/SintefManufacturing/python-urx/blob/master/urx/urrobot.py
# Must be run with library from the repo
import urx
import logging

tcp_host_ip = "10.75.15.94"

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)

    rob = urx.Robot(tcp_host_ip)
    #rob = urx.Robot("localhost")
    rob.set_tcp((0, 0, 0, 0, 0, 0))
    rob.set_payload(0.5, (0, 0, 0))
    try:
        l = 0.05
        v = 0.05
        a = 0.3
        pose = rob.getl()
        print("robot tcp is at: ", pose)
        print("absolute move in base coordinate ")
        pose[2] += l
        rob.movel(pose, acc=a, vel=v)
        print("relative move in base coordinate ")
        rob.translate((0, 0, -l), acc=a, vel=v)
        print("relative move back and forth in tool coordinate")
        rob.translate_tool((0, 0, -l), acc=a, vel=v)
        rob.translate_tool((0, 0, l), acc=a, vel=v)
    finally:
        rob.close()
