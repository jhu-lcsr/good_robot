"""
"""
import ursecmon

__author__ = "Olivier Roulet-Dubonnet"
__copyright__ = "Copyright 2011-2015, Sintef Raufoss Manufacturing"
__license__ = "LGPLv3"


class URcomm(object):
    def __init__(self, host): 

        # use_rt=False, use_simulation=False):
        # self.host = host
        # self.csys = None
        self.logger = logging.getLogger("urx")

        self.logger.debug("Opening secondary monitor socket")
        self.secmon = ursecmon.SecondaryMonitor(host)
    
        self.workspace_limits = workspace_limits
        self.moveto_limits = (
            [[0.300, 0.600], [-0.250, 0.180], [0.195, 0.571]])

        # Tool pose tolerance for blocking calls
        self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]
        self.socket_name = "gripper_socket"


        # Joint tolerance for blocking calls
        self.joint_tolerance = 0.01

        self.socket_open_str = '\tsocket_open("127.0.0.1", 63352, "gripper_socket")\n'
        self.socket_close_str = '\tsocket_close("gripper_socket")\n'

    # -- Gripper commands

    def activate_gripper(self):
        prog = "def actGrip():\n"
        # Activate gripper
        prog += 'socket_open(\"{}\",{},\"{}\"\n)'.format("127.0.0.1",
                                                         63352,
                                                         self.socket_name)
        # TODO: does this cause the gripper to open and close? to acitvate
        prog += "socket_set_var(\"{}\",{},\"{}\")\n".format("ACT", 1,
                                                            self.socket_name)
        prog += "socket_set_var(\"{}\",{},\"{}\")\n".format("GTO", 1,
                                                            self.socket_name)
        prog += "end\n"

    # We also talk to Robotiq 2F-85 gripper through the UR5 "API"

    def open_gripper(self, async=False):
        # print("!-- open gripper")
        prog = "def openGrip():\n"
        prog += self.socket_close_str
        prog += self.socket_open_str
        prog += "\tsocket_set_var(\"{}\",{},\"{}\")\n".format("POS", 0,
                                                         self.socket_name)
        prog += "end\n"
        self.send_program(prog)

    def close_gripper(self, async=False):
        # print("!-- close gripper")
        prog = "def closeGrip():\n"
        prog += self.socket_close_str
        prog += self.socket_open_str
        prog += "\tsocket_set_var(\"{}\",{},\"{}\")\n".format("POS", 1,
                                                         self.socket_name)
        prog += "end\n"
        self.send_program(prog)

        gripper_fully_closed = self.check_grasp()
        return gripper_fully_closed

    def check_grasp(self):
        prog = "def setAnalogOutToGripPos():\n"
        prog += self.socket_close_str
        prog += self.socket_open_str
        prog += '\trq_pos = socket_get_var("POS","gripper_socket")\n'
        prog += "\tset_standard_analog_out(0, rq_pos / 255)\n"
        prog += "end\n"
        self.send_program(prog)
        # TODO: do I need a slight delay here?

        tool_pos = self.get_state('tool_data')

        return tool_pos > 9 # TODO

    # -- Data commands

    def get_state(self):
        parse_functions = {'joint_data': get_joint_data, 'cartesian_info':
                           get_cartesian_info, 'tool_data': get_tool_data}

        def get_joint_data(self, _log=True):
            jts = self.secmon.get_joint_data()
            joint_positions = [jts["q_actual0"], jts["q_actual1"],
                               jts["q_actual2"], jts["q_actual3"],
                               jts["q_actual4"], jts["q_actual5"]] 
            if _log:
                self.logger.debug("Received joint data from robot: %s",
                                  joint_positions)
            return joint_positions

        def get_cartesian_info(self, _log=True):
            pose = self.secmon.get_cartesian_info()
            if pose:
                pose = [pose["X"], pose["Y"], pose["Z"],
                        pose["Rx"], pose["Ry"], pose["Rz"]]
            if _log:
                self.logger.debug("Received pose from robot: %s", pose)
            return pose

        def get_tool_data():
            return self.secmon.get_analog_out(0) # TODO: is this a value b/tw 0 and 10?

        return parse_functions[subpackage]() # cute trick


    def _send_program(self, prog):
        # mostly adding a printout for ease of debugging
        self.logger.info("Sending program: " + prog)
        self.secmon.send_program(prog)

    # -- Utils

    def _btw(self, a, min, max):
        if (a >= min) and (a <= max):
            return True
        return False

    def _isSafe(position, limits):
        safe = self.btw(position[0], limits[0][0], limits[0][1]) and \
                self.btw(position[1], limits[1][0], limits[1][1]) and \
                self.btw(position[2], limits[2][0], limits[2][1]):
        return safe

    def _format_move(self, command, tpose, acc, vel, radius=0, prefix=""):
        # prefix= p for position, none for joints
        # tpose = [round(i, self.max_float_length) for i in tpose]
        tpose.append(acc)
        tpose.append(vel)
        tpose.append(radius)
        return "\t{}({}[{},{},{},{},{},{}], a={}, v={}, r={})\n".format(command, prefix, *tpose)


    '''
    def _wait_for_move(self, target, threshold=None, timeout=5, joints=False):
        """ 
        wait for a move to complete. Unfortunately there is no good way to know when a move has finished
        so for every received data from robot we compute a dist equivalent and when it is lower than
        'threshold' we return.
        if threshold is not reached within timeout, an exception is raised
        """
        self.logger.debug(
            "Waiting for move completion using threshold %s and target %s", threshold, target)
        start_dist = self._get_dist(target, joints)
        if threshold is None:
            threshold = start_dist * 0.8
            if threshold < 0.001:  # roboten precision is limited
                threshold = 0.001
            self.logger.debug("No threshold set, setting it to %s", threshold)
        count = 0
        while True:
            if not self.is_running():
                raise RobotException("Robot stopped")
            dist = self._get_dist(target, joints)
            self.logger.debug(
                "distance to target is: %s, target dist is %s", dist, threshold)
            if not self.secmon.is_program_running():
                if dist < threshold:
                    self.logger.debug(
                        "we are threshold(%s) close to target, move has ended", threshold)
                    return
                count += 1
                if count > timeout * 10:
                    raise RobotException("Goal not reached but no program has been running for {} seconds. dist is {}, threshold is {}, target is {}, current pose is {}".format(
                        timeout, dist, threshold, target, URRobot.getl(self)))
            else:
                count = 0

    def _get_dist(self, target, joints=False):
        if joints:
            return self._get_joints_dist(target)
        else:
            return self._get_lin_dist(target)

    def _get_lin_dist(self, target):
        # FIXME: we have an issue here, it seems sometimes the axis angle received from robot
        pose = URRobot.getl(self, wait=True)
        dist = 0
        for i in range(3):
            dist += (target[i] - pose[i]) ** 2
        for i in range(3, 6):
            dist += ((target[i] - pose[i]) / 5) ** 2  # arbitraty length like
        return dist ** 0.5

    def _get_joints_dist(self, target):
        joints = self.getj(wait=True)
        dist = 0
        for i in range(6):
            dist += (target[i] - joints[i]) ** 2
    '''

    # -- Move commands 

    def move_to(self, position, orientation, acc=self.acc_vel,
                vel=self.joint_vel, radius=0, wait=True):
        # position ins meters, orientation is axis-angle
        if _isSafe(position, self.moveto_limits):
            prog = "def moveTo():\n"
            # t = 0, r = radius
            if orientation is None: 
                self.logger.debug("Attempting to move position but not orientation")
                orientation = self.get_state('cartesian_info')[3:]

            prog += self._format_move("movel", np.concatenate((position, orientation)),
                                      acc=acc, vel=vel, prefix="p")
            prog += "end\n"
            self.send_program(prog)
        else:
            self.logger.debug("NOT Safe. NOT moving to: %s, due to LIMITS: %s",
                                  position, self.moveto_limits)

    def move_joints(self, joint_configuration, wait=True):
        # specified in radians

        if self.is_sim:
            pass

        prog = "def moveJoint():\n"
        prog += self._format_move("movel", joint_configuration,
                                  acc=acc, vel=vel, prefix="")
        prog += "end\n"
        self.send_program(prog)
        # if wait:
            # self._wait_for_move(tpose[:6], threshold=threshold)
        #     return self.getl()


    def go_home(self):
        self.logger.debug("Going home.")
        self.move_joints(self.home_joint_config)

    def combo_move(self, pose_list, wait=True):
        """
        Example use:
        # end_position = ['p', 0.597, 0.000, 0.550]
        # end_axisangle = [2.18, -2.35, 2.21]
        # end_pose = np.concatenate((end_position, end_axisangle))
        # throw_pose_list = [start_pose, middle_pose, "open", end_pose]
        """
        acc, vel, radius = 1, 1, 0.3
        prog = "def combo_move():\n"
        prog += self.socket_close_str
        prog += self.socket_open_str
        for idx, pose in enumerate(pose_list):
            if idx == (len(pose_list) - 1):
                radius = 0.01
            if str(pose) == 'open':
                msg = "socket_set_var(\"{}\",{},\"{}\")\n".format("POS", 0,
                                                                  self.socket_name)
            else:
                # WARNING: this does not have safety checks!
                if str(pose[0]) == 'j':
                    prog += self._format_move(
                        "movej", pose[1:], acc, vel, radius, prefix="") + "\n"
                elif str(pose[0]) == 'p':
                    prog += self._format_move(
                        'movej', pose[1:], acc, vel, radius, prefix="p") + "\n"
        prog += "end\n"
        self.send_program(prog)

        if wait:
            self._wait_for_move(target=pose_list[-1][1:], threshold=threshold)
            return self.getl()

        """ this stops between points
        print('throw acc will be', self.joint_acc * 1)  # 4)
        print('throw vel will be', self.joint_vel * 1)  # 0)
        self.move_to(start_position, start_axisangle, acc_scaling=K,
                     vel_scaling=K, radius=0)  # last # is blend radius
        # , acc_scaling=K, vel_scaling=K, radius=0)  # last # is blend radius
        self.move_joints(curled_config)
        self.move_to(end_position, end_axisangle, acc_scaling=K,
                     vel_scaling=K, radius=0.5)  # last # is blend radius
        # gripper.open_gripper()
        self.move_to(np.array(end_position) - np.array((0.020, 0, -0.020)), end_axisangle, acc_scaling=K,
                     vel_scaling=K, radius=0.1)  # last # is blend radius
        self.move_to(start_position, start_axisangle, acc_scaling=K,
                     vel_scaling=K, radius=0)  # last # is blend radius
        """

        '''
        tcp_command = "def throw_traj():\n"
        # start
        # tcp_command += "movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % \
        # (start_position[0], start_position[1], start_position[2],
        # start_axisangle[0], start_axisangle[1], start_axisangle[2],
        # self.joint_acc * K, self.joint_vel * K, blend_radius)
        # # curl
        tcp_command += " movej([%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % \
            (curled_config[0], curled_config[1], curled_config[2],
             curled_config[3], curled_config[4], curled_config[5],
             self.joint_acc * K, self.joint_vel * K, 0)
        # unwind
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % \
            (end_position[0], end_position[1], end_position[2],
             end_axisangle[0], end_axisangle[1], end_axisangle[2],
             self.joint_acc * K, self.joint_vel * K, blend_radius * 0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % \
            (end_position[0] - 0.020, end_position[1], end_position[2]+0.030,
             end_axisangle[0], end_axisangle[1], end_axisangle[2],
             self.joint_acc * K, self.joint_vel * K, blend_radius * 0.3)
        # go home
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.0)\n" % \
            (start_position[0], start_position[1], start_position[2],
             start_axisangle[0], start_axisangle[1], start_axisangle[2],
             self.joint_acc * K * 0.1, self.joint_vel * K * 0.1)
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()
        '''

        # hardcoded open gripper (close to 3/4 of unwind, b/f deccel phase)
        # time.sleep(1.25)
        # self.open_gripper()
        # time.sleep(2)

        # Pre-compute blend radius
        # blend_radius = min(abs(bin_position[1] - position[1])/2 - 0.01, 0.2)
        # tcp_command += "movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % \
        # (position[0], position[1], bin_position[2],
        # tool_orientation[0], tool_orientation[1], 0.0,
        # self.joint_acc, self.joint_vel, blend_radius)

'''
    def get_state(self):
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect((self.tcp_host_ip, self.tcp_port))

       state_data = tcp_socket.recv(1024)
        # Read package header

        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!iB", data_bytes[0:5])[0]
        robot_message_type = struct.unpack("!iB", data_bytes[0:5])[1]

        # print('robot message type', robot_message_type)

        while (robot_message_type != 16):
            print('keep trying')
            state_data = tcp_socket.recv(1024)
            data_bytes = bytearray()
            data_bytes.extend(state_data)
            data_length = struct.unpack("!iB", data_bytes[0:5])[0]
            robot_message_type = struct.unpack("!iB", data_bytes[0:5])[1]

        byte_idx = 5

            elif ptype == 3:
        # Parse sub-packages
        subpackage_types = {
            'joint_data': 1, 'cartesian_info': 4, 'force_mode_data': 7, 'tool_data': 2}
        while byte_idx < data_length:
            # package_length = int.from_bytes(data_bytes[byte_idx:(byte_idx+4)], byteorder='big', signed=False)
            package_length = struct.unpack(
                "!i", data_bytes[byte_idx:(byte_idx+4)])[0]
            byte_idx += 4
            package_idx = data_bytes[byte_idx]
            if package_idx == subpackage_types[subpackage]:
                byte_idx += 1
                break
            byte_idx += package_length - 4
        tcp_socket.close()
'''
'''
elif ptype == 4:
        allData["CartesianInfo"] = self._get_data(pdata, "iBdddddd", ("size", "type", "X", "Y", "Z", "Rx", "Ry", "Rz"))

elif ptype == 2:
       allData["ToolData"] = self._get_data(pdata, "iBbbddfBffB", ("size", "type", "analoginputRange2", "analoginputRange3", "analogInput2", "analogInput3", "toolVoltage48V", "toolOutputVoltage", "toolCurrent", "toolTemperature", "toolMode"))
        
elif ptype == 3:
        fmt = "iBhhbbddbbddffffBBb"     # firmware < 3.0
        allData["MasterBoardData"] = self._get_data(pdata, fmt, ("size", "type", "digitalInputBits", "digitalOutputBits", "analogInputRange0", "analogInputRange1", "analogInput0", "analogInput1", "analogInputDomain0", "analogInputDomain1", "analogOutput0", "analogOutput1", "masterBoardTemperature", "robotVoltage48V", "robotCurrent", "masterIOCurrent"))  # , "masterSafetyState" ,"masterOnOffState", "euromap67InterfaceInstalled"   ))
'''
