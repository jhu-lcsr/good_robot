"""
"""
import ursecmon
import logging
import numpy as np

__author__ = "Olivier Roulet-Dubonnet"
__copyright__ = "Copyright 2011-2015, Sintef Raufoss Manufacturing"
__license__ = "LGPLv3"


class URcomm(object):
    def __init__(self, host, joint_vel, joint_acc, home_joint_config=None):

        self.joint_vel = joint_vel
        self.joint_acc = joint_acc

        # use_rt=False, use_simulation=False):
        # self.host = host
        # self.csys = None
        self.logger = logging.getLogger("urx")
        self.logger.debug("Opening secondary monitor socket")

        self.secmon = ursecmon.SecondaryMonitor(host)

        # NOTE: this is for throw practice
        if home_joint_config is None:
            home_in_deg = np.array(
                # [-107, -105, 130, -92, -44, -30]) * 1.0  # sideways # bent wrist 1 2
                [-197, -105, 130, -110, -90, -30]) * 1.0
            # [-107, -105, 130, -110, -90, -30]) * 1.0  # sideways
            # [-107, -105, 130, -85, -90, -30]) * 1.0  # sideways bent wrist
            self.home_joint_config = np.deg2rad(home_in_deg)
        else:
            self.home_joint_config = home_joint_config
        self.logger.debug("Home config: ", self.home_joint_config)

        self.moveto_limits = (
            [[0.300, 0.600], [-0.250, 0.180], [0.195, 0.571]])

        # Tool pose tolerance for blocking calls (meters)
        # HACK lower tolerance for now 31 July,bent wrist move not completing
        # self.pose_tolerance = [0.002, 0.002, 0.002, 0.010, 0.010, 0.010]
        self.pose_tolerance = [0.005, 0.005, 0.005, 0.020, 0.020, 0.020]

        self.socket_name = "gripper_socket"

        self.socket_open_str = '\tsocket_open("127.0.0.1", 63352, "gripper_socket")\n'
        self.socket_close_str = '\tsocket_close("gripper_socket")\n'

        """
        FOR is the variable
        range is 0 - 255
        0 is no force
        255 is full force
        """
        """
        SPE is the variable
        range is 0 - 255
        0 is no speed
        255 is full speed
        """
        """
        POS is the variable
        range is 0 - 255
        0 is open
        255 is closed
        """
        self.max_float_length = 6  # according to python-urx lib, UR may have max float length

    # -- Gripper commands

    def activate_gripper(self):
        prog = "def actGrip():\n"
        # Activate gripper
        prog += self.socket_close_str
        prog += self.socket_open_str
        # TODO: does this cause the gripper to open and close? to acitvate
        prog += "socket_set_var(\"{}\",{},\"{}\")\n".format("ACT", 1,
                                                            self.socket_name)
        prog += "socket_set_var(\"{}\",{},\"{}\")\n".format("GTO", 1,
                                                            self.socket_name)
        prog += "end\n"
        self.logger.debug("Activating gripper")
        self.send_program(prog)

    # We also talk to Robotiq 2F-85 gripper through the UR5 "API"

    def open_gripper(self, async=False):
        prog = "def openGrip():\n"
        prog += self.socket_close_str
        prog += self.socket_open_str
        prog += "\tsocket_set_var(\"{}\",{},\"{}\")\n".format("SPE", 255,
                                                              self.socket_name)
        prog += "\tsocket_set_var(\"{}\",{},\"{}\")\n".format("POS", 0,
                                                              self.socket_name)
        prog += "end\n"
        self.logger.debug("opening gripper")
        self.send_program(prog)

    def close_gripper(self, async=False):
        # print("!-- close gripper")
        prog = "def closeGrip():\n"
        prog += self.socket_close_str
        prog += self.socket_open_str
        prog += "\tsocket_set_var(\"{}\",{},\"{}\")\n".format("FOR", 20,
                                                              self.socket_name)
        prog += "\tsocket_set_var(\"{}\",{},\"{}\")\n".format("SPE", 255,
                                                              self.socket_name)
        prog += "\tsocket_set_var(\"{}\",{},\"{}\")\n".format("POS", 255,
                                                              self.socket_name)
        prog += "end\n"
        self.send_program(prog)
        self.logger.debug("Closing gripper")

        # gripper_fully_closed = self.check_grasp()
        gripper_fully_closed = True
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

        return tool_pos > 9  # TODO

    # -- Data commands

    def get_state(self, subpackage):

        def get_joint_data(_log=True):
            jts = self.secmon.get_joint_data()
            joint_positions = [jts["q_actual0"], jts["q_actual1"],
                               jts["q_actual2"], jts["q_actual3"],
                               jts["q_actual4"], jts["q_actual5"]]
            if _log:
                self.logger.debug("Received joint data from robot: %s",
                                  joint_positions)
            return joint_positions

        def get_cartesian_info(_log=True):
            pose = self.secmon.get_cartesian_info()
            if pose:
                pose = [pose["X"], pose["Y"], pose["Z"],
                        pose["Rx"], pose["Ry"], pose["Rz"]]
            if _log:
                self.logger.debug("Received pose from robot: %s", pose)
            return pose

        def get_tool_data():
            # TODO: is this a value b/tw 0 and 10?
            return self.secmon.get_analog_out(0)

        parse_functions = {'joint_data': get_joint_data, 'cartesian_info':
                           get_cartesian_info, 'tool_data': get_tool_data}
        return parse_functions[subpackage]()  # cute trick

    def send_program(self, prog, is_sim=False):
        # mostly adding a printout for ease of debugging
        if not is_sim:
            self.logger.info("Sending program: " + prog)
            self.secmon.send_program(prog)
        else:
            self.logger.info("SIM. Would have sent program: " + prog)

    # -- Utils

    def _btw(self, a, min, max):
        if (a >= min) and (a <= max):
            return True
        return False

    def _is_safe(position, limits):
        safe = self.btw(position[0], limits[0][0], limits[0][1]) and \
            self.btw(position[1], limits[1][0], limits[1][1]) and \
            self.btw(position[2], limits[2][0], limits[2][1])
        return safe

    def _format_move(self, command, tpose, acc, vel, radius=0, time=0, prefix=""):
        # prefix= p for position, none for joints
        tpose = [round(i, self.max_float_length) for i in tpose]
        # can i specifiy time?
        tpose.append(acc)
        tpose.append(vel)
        tpose.append(radius)
        tpose.append(time)
        return "\t{}({}[{}, {}, {}, {}, {}, {}], a={}, v={}, r={}, t={})\n".format(command, prefix, *tpose)

   # tcp_command += " set_digital_out(8,False)\n"
   # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" \
   # % (position[0], position[1], position[2] + 0.1, tool_orientation[0],
   # tool_orientation[1], 0.0, self.joint_acc * 0.5, self.joint_vel * 0.5)
   # tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" \
   # % (position[0], position[1], position[2], tool_orientation[0],
   # tool_orientation[1], 0.0, self.joint_acc * 0.1, self.joint_vel * 0.1)
   # tcp_command += " set_digital_out(8,True)\n"

    # -- Move commands

    def move_to(self, position, orientation, vel=None, acc=None, radius=0, wait=True):
        if vel is None:
            vel = self.joint_vel
        if acc is None:
            acc = self.joint_acc
        # position ins meters, orientation is axis-angle
        if _is_safe(position, self.moveto_limits):
            prog = "def moveTo():\n"
            # t = 0, r = radius
            if orientation is None:
                self.logger.debug(
                    "Attempting to move position but keep orientation")
                orientation = self.get_state('cartesian_info')[3:]

            prog += self._format_move("movel", np.concatenate((position, orientation)),
                                      acc=acc, vel=vel, prefix="p")
            prog += "end\n"
            self.send_program(prog)
        else:
            self.logger.debug("NOT Safe. NOT moving to: %s, due to LIMITS: %s",
                              position, self.moveto_limits)
        if wait:
            self._wait_for_move(np.concatenate((position, orientation)),
                                joints=False)

    def move_joints(self, joint_configuration, vel=None, acc=None, wait=True):
        if vel is None:
            vel = self.joint_vel
        if acc is None:
            acc = self.joint_acc

        # specified in radians
        prog = "def moveJoint():\n"
        prog += self._format_move("movel", joint_configuration,
                                  vel=vel, acc=acc, prefix="")
        prog += "end\n"
        self.send_program(prog)
        if wait:
            self._wait_for_move(joint_configuration, joints=True)

    def go_home(self):
        self.logger.debug("Going home.")
        self.move_joints(self.home_joint_config)

    def combo_move(self, moves_list, wait=True, is_sim=False):
        """
        Example use:
        pose_list = [ {type:p, vel:0.1, acc:0.1, radius:0.2},
                    {type: open}]
        """
        prog = "def combo_move():\n"
        # prog += self.socket_close_str
        prog += self.socket_open_str

        for idx, a_move in enumerate(moves_list):

            if a_move["type"] == 'open':
                prog += "\tsocket_set_var(\"{}\",{},\"{}\")\n".format("SPE", 255,
                                                                      self.socket_name)
                prog += "\tsocket_set_var(\"{}\",{},\"{}\")\n".format("POS", 0,
                                                                      self.socket_name)
            else:
                acc, vel, radius = a_move["acc"], a_move["vel"], a_move["radius"]
                if radius is None:
                    radius = 0.01
                if acc is None:
                    # acc = self.joint_acc
                    acc = self.joint_acc
                if vel is None:
                    vel = self.joint_vel
                if idx == (len(moves_list) - 1):
                    radius = 0.001
                    acc = self.joint_acc
                    vel = self.joint_vel

                # WARNING: this does not have safety checks!
                if a_move["type"] == 'j':
                    prog += self._format_move(
                        "movel", a_move['pose'], acc, vel, radius, prefix="") + "\n"
                elif a_move["type"] == 'p':
                    prog += self._format_move(
                        'movel', a_move['pose'], acc, vel, radius, prefix="p") + "\n"
        prog += "end\n"
        self.send_program(prog, is_sim=is_sim)

        if wait:
            joint_flag = False
            if moves_list[-1]['type'] == 'j':
                joint_flag = True
            self._wait_for_move(target=moves_list[-1]['pose'],
                                threshold=self.pose_tolerance, joints=joint_flag)
            return self.get_state('cartesian_info')

    def throw_sideways(self, is_sim=False):
        K = 9.
        acc, vel = 1.4 * K, K

        sideways_position = np.deg2rad(np.array(
            [-107, -105, 130, -92, -44, -30]) * 1.0)  # sideways # bent wrist 1 2
        # [-107, -105, 130, -85, -90, -30]) * 1.0  # sideways bent wrist
        # sideways_position = [-0.005, 0.351, 0.245, 3.0, 0.33, 0.00]
        # sideways_position = [-0.017, 0.309,
        # 0.218, 2.6, 0.30, -0.23]  # bent wrist
        sideways_move = {'type': 'j',
                         'pose': sideways_position,
                         # 'acc': None, 'vel': None, 'radius': 0.2}
                         'acc': acc, 'vel': vel, 'radius': 0.05}

        start_position = [0.350, 0.000, 0.250, 2.12, -2.21, -0.009]
        start_move = {'type': 'p',
                      'pose': start_position,
                      # 'acc': None, 'vel': None, 'radius': 0.2}
                      'acc': acc/3., 'vel': vel/3., 'radius': 0.1}

        curled_position = [0.350, 0.000, 0.250, 1.75, -1.80, -0.62]
        curled_move = {'type': 'p',
                       'pose': curled_position,
                       'acc': acc, 'vel': vel, 'radius': 0.001}

        throw_position = [0.597, 0.000, 0.640, 2.26, -2.35, 2.24]
        # throw_position = [0.567, 0.000, 0.580, 2.38, -2.37, 1.60]
        throw_move = {'type': 'p',
                      'pose': throw_position,
                      'acc': acc, 'vel': vel, 'radius': 0.200}

        home_position = np.array(start_position) + \
            np.array([0, 0, 0.070, 0, 0, 0])
        home_position = home_position.tolist()
        home_move = {'type': 'p',
                     'pose': home_position,
                     'acc': acc/3., 'vel': vel/3., 'radius': 0.015}

        gripper_open = {'type': 'open'}

        # NOTE: important
        throw_pose_list = [throw_move,  # throw_move,
                           gripper_open, home_move, start_move, sideways_move]
        # throw_pose_list = [start_move]

        # pose_list = [start_pose, middle_pose, end_pose, start_pose]
        self.combo_move(throw_pose_list, wait=True, is_sim=is_sim)

    def throw_andy(self, wait=True, is_sim=False):
        default_jacc = 8.  # 8.0
        default_jvel = 3.0  # 3.0
        toss_jacc = 25.  # 25.0
        toss_jvel = 3.2  # 3.2
        pretoss_jconf = np.asarray(
            [90., -45., 90., -098.9, -90., 0.])*np.pi/180.0
        posttoss_jconf = np.asarray(
            # [90., -057.8, 035.1, -142.1, -90., 0.])*np.pi/180.0
            # JOINT 3 IS THE PROBLEM
            [90., -057.8, 035.1, -142.1, -90., 0.])*np.pi/180.0
        # [90., -057.8, 90, -142.1, -90., 0.])*np.pi/180.0
        pretoss_blend_radius = 0.09
        # toss_blend_radius = 0.7 # BLEND FAIL
        # toss_blend_radius = 0.6 # CAUSES CRUNCH SOUND ON WAY UP
        # toss_blend_radius = 0.005 # FINE
        toss_blend_radius = 0.6  # FINE
        # toss_blend_radius = 0.5

        tcp_msg = "def process():\n"
        tcp_msg += '    socket_open("127.0.0.1",63352,"gripper_socket")\n'
        tcp_msg += "    sync()\n"
        # tcp_msg += self._format_move("movej", pretoss_jconf, default_jacc,
        # default_jvel, pretoss_blend_radius, time=0, prefix="") + "\n"
        # tcp_msg += self._format_move("movej", posttoss_jconf, toss_jacc,
        # toss_jvel, toss_blend_radius, time=0, prefix="") + "\n"
        tcp_msg += '    movej([%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0.0,r=%f)\n' % (pretoss_jconf[0], pretoss_jconf[1], pretoss_jconf[2],
                                                                              pretoss_jconf[3], pretoss_jconf[4], pretoss_jconf[5], default_jacc, default_jvel, pretoss_blend_radius)
        tcp_msg += '    movej([%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0.0,r=%f)\n' % (posttoss_jconf[0], posttoss_jconf[1],
                                                                              posttoss_jconf[2], posttoss_jconf[3], posttoss_jconf[4], posttoss_jconf[5], toss_jacc, toss_jvel, toss_blend_radius)
        # tcp_msg += '    set_digital_out(8,False)\n'  # for RG2 gripper
        tcp_msg += "    socket_set_var(\"{}\",{},\"{}\")\n".format("SPE", 255,
                                                                   self.socket_name)
        tcp_msg += "    socket_set_var(\"{}\",{},\"{}\")\n".format("POS", 0,
                                                                   self.socket_name)
        # tcp_msg += "    sync()\n"
        # tcp_msg += self._format_move("movej", pretoss_jconf, default_jacc,
        # default_jvel, radius=0, time=0, prefix="") + "\n"
        tcp_msg += '    movej([%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0.0,r=0.0)\n' % (pretoss_jconf[0], pretoss_jconf[1],
                                                                               pretoss_jconf[2], pretoss_jconf[3], pretoss_jconf[4], pretoss_jconf[5], default_jacc, default_jvel)
        tcp_msg += '    socket_close("gripper_socket")\n'
        tcp_msg += 'end\n'
        self.send_program(tcp_msg, is_sim=is_sim)

        if wait:
            joint_flag = True
            self._wait_for_move(target=pretoss_jconf,
                                threshold=self.pose_tolerance, joints=joint_flag)
            return self.get_state('cartesian_info')

        print('done with toss')
        # self.send_program(prog, is_sim=is_sim)

    def throw(self, is_sim=False):
        self.close_gripper()
        # currently hard coded positions
        # acc, vel = 1.4, 1.05 # Safe
        # acc, vel = 8, 3 # Default
        # acc, vel = 15, 10
        # K = 8.5
        K = 2
        acc, vel = 1.4 * K, K
        start_position = [0.350, 0.000, 0.250, 2.12, -2.21, -0.009]
        start_move = {'type': 'p',
                      'pose': start_position,
                      # 'acc': None, 'vel': None, 'radius': 0.2}
                      'acc': acc, 'vel': vel, 'radius': 0.001}

        # curled_position = [0.350, 0.000, 0.250, 1.75, -1.80, -0.62]
        # curled_move = {'type': 'p',
        # 'pose': curled_position,
        # 'acc': acc, 'vel': vel, 'radius': 0.000}

        # throw_position = [0.597, 0.000, 0.640, 2.26, -2.35, 2.24]
        throw_position = [0.550, 0.000, 0.550, 2.26, -2.35, 2.24]
        # throw_position = [0.567, 0.000, 0.580, 2.38, -2.37, 1.60]
        throw_move = {'type': 'p',
                      'pose': throw_position,
                      'acc': acc, 'vel': vel, 'radius': 0.150}

        throw2_position = np.array(throw_position) + \
            np.array([0.050, 0.000, 0.105, 0, 0, 0])
        throw2_position = throw2_position.tolist()
        throw2_move = {'type': 'p',
                       'pose': throw2_position,
                       'acc': acc, 'vel': vel, 'radius': 0.01}  # max =0.05*1.4
        # return_position = [0.590, 0.000, 0.620, 2.38, -2.37, 1.60]
        # return_move = {'type': 'p',
        # 'pose': return_position,
        # 'acc': 3.5, 'vel': 3.5, 'radius': 0.100}

        home_position = np.array(start_position) + \
            np.array([0.010, 0, 0.210, 0, 0, 0])
        home_position = home_position.tolist()
        home_move = {'type': 'p',
                     'pose': home_position,
                     'acc': 1.0, 'vel': 1.0, 'radius': 0.001}

        gripper_open = {'type': 'open'}

        # throw_pose_list = [curled_move, throw_move,  # throw_move,
        throw_pose_list = [throw_move, gripper_open, throw2_move,
                           home_move, start_move]
        # throw_pose_list = [start_move]

        self.combo_move(throw_pose_list, wait=True, is_sim=is_sim)

        '''
        # Pre-compute blend radius
        # blend_radius = min(abs(bin_position[1] - position[1])/2 - 0.01, 0.2)
        # tcp_command += "movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % \
        # (position[0], position[1], bin_position[2],
        # tool_orientation[0], tool_orientation[1], 0.0,
        # self.joint_acc, self.joint_vel, blend_radius)
        '''

    def is_running(self):
        """
        Return True if robot is running (not
        necessary running a program, it might be idle)
        """
        return self.secmon.running

    def _wait_for_move(self, target, threshold=None, joints=False):
        """
        Wait for a move to complete. Unfortunately there is no good way to know
        when a move has finished so for every received data from robot we
        compute a dist equivalent and when it is lower than 'threshold' we
        return.
        if threshold is not reached within timeout, an exception is raised
        """
        self.logger.debug(
            "Waiting for move completion using threshold %s and target %s", threshold, target)
        if threshold is None:
            # threshold = [0.001] * 6
            threshold = self.pose_tolerance
            self.logger.debug("No threshold set, setting it to %s", threshold)
        while True:
            if not self.is_running():
                # raise RobotException("Robot stopped")
                self.logger.exception("ROBOT STOPPED!")
            if joints:
                actual_pose = self.get_state('joint_data')
            else:
                actual_pose = self.get_state('cartesian_info')

            dist = [np.abs(actual_pose[j] - target[j]) for j in range(6)]
            self.logger.debug(
                "distance to target is: %s, target dist is %s", dist, threshold)
            if all([np.abs(actual_pose[j] - target[j]) < self.pose_tolerance[j] for j in range(6)]):
                self.logger.debug(
                    "We are threshold(%s) close to target, move has ended", threshold)
                return

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
    '''

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
