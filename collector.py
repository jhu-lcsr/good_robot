'''
Originally from https://github.com/jhu-lcsr/costar_plan/blob/d469d62d72cd405ed07b10c62eb24391c0af1975/ctp_integration/python/ctp_integration/collector.py
'''
import matplotlib.pyplot as plt
import numpy as np
import PyKDL as kdl
import rospy
import tf2_ros as tf2
import tf_conversions.posemath as pm
import io
import message_filters

from costar_models.datasets.npz import NpzDataset
from costar_models.datasets.h5f import H5fDataset
from costar_models.datasets.image import GetJpeg
from costar_models.datasets.image import GetPng
from costar_models.datasets.image import JpegToNumpy
from costar_models.datasets.image import ConvertImageListToNumpy
from costar_models.datasets.depth_image_encoding import FloatArrayToRgbImage

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from robotiq_c_model_control.msg import CModel_robot_input as GripperMsg
import cv2

import six
import json
import sys
import datetime
from constants import GetHomeJointSpace
from constants import GetHomePose
from threading import Lock
# TODO(ahundt) move all direct h5py code back to H5fDataset class
import h5py
from ctp_integration.ros_geometry import pose_to_vec_quat_pair
from ctp_integration.ros_geometry import pose_to_vec_quat_list

def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Apply a timestamp to the front of a filename description.

    see: http://stackoverflow.com/a/5215012/99379
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


class DataCollector(object):
    '''
    Buffers data from an example then writes it to disk.

    Data received includes:
    - images from camera
    - depth data (optional)
    - current end effector pose
    - current joint states
    - current gripper status
    '''

    def __init__(
            self, robot_config,
            task,
            data_type="h5f",
            rate=10,
            data_root=".",
            img_shape=(128, 128),
            camera_frame="camera_link",
            tf_buffer=None,
            tf_listener=None,
            action_labels_to_always_log=None,
            verbose=0,
            synchronize=False):
        """ Initialize a data collector object for writing ros topic information and data collection state to disk

        img_shape: currently ignored
        camera_frame: ros tf rigid body transform frame to use as the base of the camera coodinate systems
        tf_buffer: temp store for 3d rigid body transforms, used during logging
        tf_listener: integrates with tf_buffer
        action_labels_to_always_log: 'move_to_home' is always logged by default, others can be added. This option may not work yet.
        verbose: print lots of extra info, useful for debuggging
        synchronize: will attempt to synchronize image data by timestamp. Not yet working as of 2018-05-05.
        """

        self.js_topic = "joint_states"
        # http://wiki.ros.org/depth_image_proc
        # http://www.ros.org/reps/rep-0118.html
        # http://wiki.ros.org/rgbd_launch
        # we will be getting 16 bit integer values in milimeters
        self.rgb_topic = "/camera/rgb/image_rect_color"
        # raw means it is in the format provided by the openi drivers, 16 bit int
        self.depth_topic = "/camera/depth_registered/hw_registered/image_rect"
        self.ee = "endpoint"
        self.base_link = "base_link"
        self.description = "/robot_description"
        self.data_types = ["h5f", "npz"]
        self.info_topic = "/costar/info"
        self.object_topic = "/costar/SmartMove/object"
        self.gripper_topic = "/CModelRobotInput"
        self.camera_depth_info_topic = "/camera/rgb/camera_info"
        self.camera_rgb_info_topic = "/camera/depth_registered/camera_info"
        self.camera_rgb_optical_frame = "camera_rgb_optical_frame"
        self.camera_depth_optical_frame = "camera_depth_optical_frame"
        self.verbose = verbose
        self.mutex = Lock()
        if action_labels_to_always_log is None:
            self.action_labels_to_always_log = ['move_to_home']
        else:
            self.action_labels_to_always_log = action_labels_to_always_log

        '''
        Set up the writer (to save trials to disk) and subscribers (to process
        input from ROS and store the current state).
        '''
        if tf_buffer is None:
            self.tf_buffer = tf2.Buffer()
        else:
            self.tf_buffer = tf_buffer
        if tf_listener is None:
            self.tf_listener = tf2.TransformListener(self.tf_buffer)
        else:
            self.tf_listener = tf_listener

        if isinstance(rate, int) or isinstance(rate, float):
            self.rate = rospy.Rate(rate)
        elif isinstance(rate, rospy.Rate):
            self.rate = rate
        else:
            raise RuntimeError("rate data type not supported: %s" % type(rate))

        self.root = data_root
        self.data_type = data_type
        rospy.logwarn("Dataset root set to " + str(self.root))
        if self.data_type == "h5f":
            self.writer = H5fDataset(self.root)
        elif self.data_type == "npz":
            self.writer = NpzDataset(self.root)
        else:
            raise RuntimeError("data type %s not supported" % data_type)

        self.T_world_ee = None
        self.T_world_camera = None
        self.camera_frame = camera_frame
        self.ee_frame = robot_config['end_link']
        self.rgb_time = None

        self.q = None
        self.dq = None
        self.pc = None
        self.camera_depth_info = None
        self.camera_rgb_info = None
        self.depth_img = None
        self.rgb_img = None
        self.gripper_msg = None

        self._bridge = CvBridge()
        self.task = task
        self.reset()

        if synchronize:
            # TODO(ahundt) synchronize image time stamps, consider including joint info too
            # http://docs.ros.org/kinetic/api/message_filters/html/python/
            # http://library.isr.ist.utl.pt/docs/roswiki/message_filters.html
            # may want to consider approx:
            # http://wiki.ros.org/message_filters/ApproximateTime
            # self._camera_depth_info_sub = rospy.Subscriber(self.camera_depth_info_topic, CameraInfo)
            # self._camera_rgb_info_sub = rospy.Subscriber(self.camera_rgb_info_topic, CameraInfo)
            # ensure synced data has headers: https://answers.ros.org/question/206650/subcribe-to-multiple-topics-with-message_filters/
            # example code:
            # https://github.com/gt-ros-pkg/hrl/blob/df47c6fc4fbd32df44df0060643e94cdf5741ff3/hai_sandbox/src/hai_sandbox/kinect_fpfh.py
            # https://github.com/b2256/catkin_ws/blob/fef8bc05f34262083f02e06b1585f2170d6de5a3/src/bag2orb/src/afl_sync_node_16.py
            rospy.loginfo('synchronizing data for logging')
            self._camera_depth_info_sub = rospy.Subscriber(self.camera_depth_info_topic, CameraInfo, self._depthInfoCb)
            self._camera_rgb_info_sub = rospy.Subscriber(self.camera_rgb_info_topic, CameraInfo, self._rgbInfoCb)
            self._rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
            self._depth_sub = message_filters.Subscriber(self.depth_topic, Image)
            self._time_sync_rgbd_sub = message_filters.TimeSynchronizer(
                [self._rgb_sub, self._depth_sub], 30)
            self._time_sync_rgbd_sub.registerCallback(self._rgbdCb)
        else:
            # just take the data as it comes rather than synchronizing
            self._camera_depth_info_sub = rospy.Subscriber(self.camera_depth_info_topic, CameraInfo, self._depthInfoCb)
            self._camera_rgb_info_sub = rospy.Subscriber(self.camera_rgb_info_topic, CameraInfo, self._rgbInfoCb)
            self._rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self._rgbCb)
            self._depth_sub = rospy.Subscriber(self.depth_topic, Image, self._depthCb)
        self._joints_sub = rospy.Subscriber(
                self.js_topic,
                JointState,
                self._jointsCb)
        self._info_sub = rospy.Subscriber(
                self.info_topic,
                String,
                self._infoCb)
        self._smartmove_object_sub = rospy.Subscriber(
                self.object_topic,
                String,
                self._objectCb)
        self._gripper_sub = rospy.Subscriber(
                self.gripper_topic,
                GripperMsg,
                self._gripperCb)

    def _rgbdCb(self, rgb_msg, depth_msg):
        if rgb_msg is None:
            rospy.logwarn("_rgbdCb: rgb_msg is None !!!!!!!!!")
        try:
            # max out at 10 hz assuming 30hz data source
            # TODO(ahundt) make mod value configurable
            if rgb_msg.header.seq % 3 == 0:
                cv_image = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")

                # decode the data, this will take some time

                rospy.loginfo('rgb color cv_image shape: ' + str(cv_image.shape) + ' depth sequence number: ' + str(msg.header.seq))
                # print('rgb color cv_image shape: ' + str(cv_image.shape))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                # encode the jpeg with high quality
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 99]
                rgb_img = cv2.imencode('.jpg', cv_image, encode_params)[1].tobytes()
                # rgb_img = GetJpeg(np.asarray(cv_image))

                cv_depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                depth_encoded_as_rgb_numpy = encode_depth_numpy(cv_depth_image)
                bytevalues = cv2.imencode('.png', depth_encoded_as_rgb_numpy)[1].tobytes()

                with self.mutex:
                    self.rgb_time = msg.header.stamp
                    self.rgb_img = rgb_img
                    # self.depth_info = depth_camera_info
                    # self.rgb_info = rgb_camera_info
                    self.depth_img_time = msg.header.stamp
                    # self.depth_img = np_image
                    # self.depth_img = img_str
                    self.depth_img = bytevalues
            #print(self.rgb_img)
        except CvBridgeError as e:
            rospy.logwarn(str(e))

    def _rgbCb(self, msg):
        if msg is None:
            rospy.logwarn("_rgbCb: msg is None !!!!!!!!!")
        try:
            # max out at 10 hz assuming 30hz data source
            if msg.header.seq % 3 == 0:
                cv_image = self._bridge.imgmsg_to_cv2(msg, "rgb8")
                # decode the data, this will take some time

                # rospy.loginfo('rgb color cv_image shape: ' + str(cv_image.shape) + ' depth sequence number: ' + str(msg.header.seq))
                # print('rgb color cv_image shape: ' + str(cv_image.shape))
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                rgb_img = cv2.imencode('.jpg', cv_image)[1].tobytes()
                # rgb_img = GetJpeg(np.asarray(cv_image))

                with self.mutex:
                    self.rgb_time = msg.header.stamp
                    self.rgb_img = rgb_img
            #print(self.rgb_img)
        except CvBridgeError as e:
            rospy.logwarn(str(e))

    def _infoCb(self, msg):
        with self.mutex:
            self.info = msg.data

    def _depthInfoCb(self, msg):
        with self.mutex:
            self.depth_info = msg

    def _rgbInfoCb(self, msg):
        with self.mutex:
            self.rgb_info = msg

    def _objectCb(self, msg):
        with self.mutex:
            self.object = msg.data

    def _gripperCb(self, msg):
        with self.mutex:
            self.gripper_msg = msg

    def _depthCb(self, msg):
        try:
            if msg.header.seq % 3 == 0:
                cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

                # ref: https://stackoverflow.com/a/25592959
                # also: https://stackoverflow.com/a/17970817
                # kinda works, but only 8 bit image....

                # img_str = cv2.imencode('.png', cv_image, cv2.CV_16U)[1].tobytes()
                # img_str = np.frombuffer(cv2.imencode('.png', cv_image)[1].tobytes(), np.uint8)
                # doesn't work
                # img_str = np.string_(cv2.imencode('.png', cv_image)[1].tostring())
                # img_str = io.BytesIO(img_str).getvalue()
                # doesn't work
                # img_str = io.BytesIO(cv2.imencode('.png', cv_image)[1].tobytes().getvalue())
                # These values are in mm according to:
                # https://github.com/ros-perception/depthimage_to_laserscan/blob/indigo-devel/include/depthimage_to_laserscan/depth_traits.h#L49
                # np_image = np.asarray(cv_image, dtype=np.uint16)

                # depth_image = PIL.Image.fromarray(np_image)

                # if depth_image.mode == 'I;16':
                #     # https://github.com/python-pillow/Pillow/issues/1099
                #     # https://github.com/arve0/leicaexperiment/blob/master/leicaexperiment/experiment.py#L560
                #     depth_image = depth_image.convert(mode='I')
                # max_val = np.max(np_image)
                # min_val = np.min(np_image)
                # print('max val: ' + str(max_val) + ' min val: ' + str(min_val))
                # decode the data, this will take some time
                # output = io.BytesIO()
                # depth_image.save(output, format="PNG")

                # begin 32 bit float code (too slow)
                # cv_image = self._bridge.imgmsg_to_cv2(msg, "32FC1")
                # # These values are in mm according to:
                # # https://github.com/ros-perception/depthimage_to_laserscan/blob/indigo-devel/include/depthimage_to_laserscan/depth_traits.h#L49
                # np_image = np.asarray(cv_image, dtype=np.float32) * 1000.0
                # # max_val = np.max(np_image)
                # # min_val = np.min(np_image)
                # # print('max val: ' + str(max_val) + ' min val: ' + str(min_val))
                # # decode the data, this will take some time
                # depth_image = FloatArrayToRgbImage(np_image)
                # output = io.BytesIO()
                # depth_image.save(output, format="PNG")
                # end 32 bit float code (too slow)

                # convert to meters from milimeters
                # plt.imshow(cv_image, cmap='nipy_spectral')
                # plt.pause(.01)
                # plt.draw()
                # print('np_image shape: ' + str(np_image.shape))

                # split into three channels
                # np_image = np.asarray(cv_image, dtype=np.uint32) * 1000
                # r = np.array(np.divide(np_image, 256*256), dtype=np.uint8)
                # g = np.array(np.mod(np.divide(np_image, 256), 256), dtype=np.uint8)
                # b = np.array(np.mod(np_image, 256), dtype=np.uint8)

                # split into two channels with a third zero channel

                # bytevalues = uint16_depth_image_to_png_numpy(cv_image)
                depth_encoded_as_rgb_numpy = encode_depth_numpy(cv_image)
                bytevalues = cv2.imencode('.png', depth_encoded_as_rgb_numpy)[1].tobytes()

                with self.mutex:
                    self.depth_img_time = msg.header.stamp
                    # self.depth_img = np_image
                    # self.depth_img = img_str
                    self.depth_img = bytevalues
                # print (self.depth_img)
        except CvBridgeError as e:
            rospy.logwarn(str(e))

    def setTask(self, task):
        self.task = task

    def reset(self):
        self.data = {}
        self.data["nsecs"] = []
        self.data["secs"] = []
        self.data["q"] = []
        self.data["dq"] = []
        self.data["pose"] = []
        self.data["camera"] = []
        self.data["image"] = []
        self.data["depth_image"] = []
        self.data["goal_idx"] = []
        self.data["gripper"] = []
        self.data["label"] = []
        self.data["info"] = []
        self.data["depth_info"] = []
        self.data["rgb_info"] = []
        self.data["object"] = []
        self.data["object_pose"] = []
        self.data["labels_to_name"] = list(self.task.labels)
        self.data["rgb_info_D"] = []
        self.data["rgb_info_K"] = []
        self.data["rgb_info_R"] = []
        self.data["rgb_info_P"] = []
        self.data["rgb_info_distortion_model"] = []
        self.data["depth_info_D"] = []
        self.data["depth_info_K"] = []
        self.data["depth_info_R"] = []
        self.data["depth_info_P"] = []
        self.data["depth_distortion_model"] = []
        self.data["all_tf2_frames_as_yaml"] = []
        self.data["all_tf2_frames_from_base_link_vec_quat_xyzxyzw_json"] = []
        self.data["visualization_marker"] = []
        self.data["camera_rgb_optical_frame_pose"] = []
        self.data["camera_depth_optical_frame_pose"] = []
        #self.data["depth"] = []

        self.info = None
        self.object = None
        self.prev_objects = []
        self.action = None
        self.prev_action = None
        self.current_ee_pose = None
        self.last_goal = 0
        self.prev_last_goal = 0
        self.home_xyz_quat = GetHomePose()

    def _jointsCb(self, msg):
        with self.mutex:
            self.q = msg.position
            self.dq = msg.velocity
        if self.verbose > 3:
            rospy.loginfo(self.q, self.dq)

    def save(self, seed, result, log=None):
        '''
        Save function that wraps dataset access, dumping the data
        which is currently buffered in memory to disk.

        seed: An integer identifer that should go in the filename.
        result: Options are 'success' 'failure' or 'error.failure'
            error.failure should indicate a failure due to factors
            outside the purposes for which you are collecting data.
            For example, if there is a network communication error
            and you are collecting data for a robot motion control
            problem, that can be considered an error based failure,
            since the goal task may essentially already be complete.
            You may also pass a numerical reward value, for numerical
            reward values, 0 will name the output file 'failure', and
            anything > 0 will name the output file 'success'.
        log: A string containing any log data you want to save,
            For example, if you want to store information about errors
            that were encounterd to help figure out if the problem
            is one that is worth including in the dataset even if it
            is an error.failure case. For example, if the robot
            was given a motion command that caused it to crash into the
            floor and hit a safety stop, this could be valuable training
            data and the error string will make it easier to determine
            what happened after the fact.
        '''
        if self.verbose:
            for k, v in self.data.items():
                print(k, np.array(v).shape)
            print(self.data["labels_to_name"])
            print("Labels and goals:")
            print(self.data["label"])
            print(self.data["goal_idx"])

        # TODO(ahundt) make nonspecific to hdf5
        # dt = h5py.special_dtype(vlen=bytes)
        # self.data['depth_image'] = np.asarray(self.data['depth_image'], dtype=dt)
        self.data['depth_image'] = np.asarray(self.data['depth_image'])
        self.data['image'] = np.asarray(self.data['image'])
        if log is None:
            # save an empty string in the log if nothing is specified
            log = ''
        self.data['log'] = np.asarray(log)

        if isinstance(result, int) or isinstance(result, float):
            result = "success" if result > 0. else "failure"

        filename = timeStamped("example%06d.%s.h5f" % (seed, result))
        rospy.loginfo('Saving dataset example with filename: ' + filename)
        # for now all examples are considered a success
        self.writer.write(self.data, filename, image_types=[("image", "jpeg"), ("depth_image", "png")])
        self.reset()

    def set_home_pose(self, pose):
        self.home_xyz_quat = pose

    def update(self, action_label, is_done):
        '''
        Compute endpoint positions and update data. Should happen at some
        fixed frequency like 10 hz.

        Parameters:
        -----------
        action: name of high level action being executed
        '''

        switched = False
        if not self.action == action_label:
            if self.action is not None:
                switched = True
            self.prev_action = self.action
            self.action = action_label
            self.prev_objects.append(self.object)
            self.object = None
        if switched or is_done:
            self.prev_last_goal = self.last_goal
            self.last_goal = len(self.data["label"])
            len_label = len(self.data["label"])

            # Count one more if this is the last frame -- since our goal could
            # not be the beginning of a new action
            if is_done:
                len_label += 1
                extra = 1
            else:
                extra = 0

            rospy.loginfo(
                "Starting new action: " +
                str(action_label) +
                ", prev was from " +
                str(self.prev_last_goal) +
                # ' ' + (str(self.data["label"][self.prev_last_goal]) if self.prev_last_goal else "") +
                " to " + str(self.last_goal)
                # ' ' + (str(self.data["label"][self.last_goal]) if self.last_goal else "") +
                )
            self.data["goal_idx"] += (self.last_goal - self.prev_last_goal + extra) * [self.last_goal]

            len_idx = len(self.data["goal_idx"])
            if not len_idx == len_label:
                rospy.logerr("lens = " + str(len_idx) + ", " + str(len_label))
                raise RuntimeError("incorrectly set goal idx")

        # action text to check will be the string contents after the colon
        label_to_check = action_label.split(':')[-1]

        should_log_this_timestep = (self.object is not None or
                                    label_to_check in self.action_labels_to_always_log)
        if not should_log_this_timestep:
            # here we check if a smartmove object is defined to determine
            # if we should be logging at this time.
            if self.verbose:
                rospy.logwarn("passing -- has not yet started executing motion")
            return True

        if self.verbose:
            rospy.loginfo("Logging: " + str(self.action) +
                    ", obj = " + str(self.object) +
                    ", prev = " + str(self.prev_objects))

        local_time = rospy.Time.now()
        # this will get the latest available time
        latest_available_time_lookup = rospy.Time(0)

        ##### BEGIN MUTEX
        with self.mutex:
            # get the time for this data sample
            if self.rgb_time is not None:
                t = self.rgb_time
            else:
                t = local_time

            self.t = t
            # make sure we keep the right rgb and depth
            img_jpeg = self.rgb_img
            depth_png = self.depth_img

        have_data = False
        # how many times have we tried to get the transforms
        attempts = 0
        max_attempts = 10
        # the number attempts that should
        # use the backup timestamps
        backup_timestamp_attempts = 4
        while not have_data:
            try:
                c_pose = self.tf_buffer.lookup_transform(self.base_link, self.camera_frame, t)
                ee_pose = self.tf_buffer.lookup_transform(self.base_link, self.ee_frame, t)
                if self.object:
                    lookup_object = False
                    # Check for the detected object at the current time.
                    try:
                        obj_pose = self.tf_buffer.lookup_transform(self.base_link, self.object, t)
                        lookup_object = True
                    except (tf2.ExtrapolationException, tf2.ConnectivityException) as e:
                        pass
                    if not lookup_object:
                        # If we can't get the current time for the object,
                        # get the latest available. This particular case will be common.
                        # This is because the object detection srcipt can only run when the
                        # arm is out of the way.
                        obj_pose = self.tf_buffer.lookup_transform(self.base_link, self.object, latest_available_time_lookup)

                rgb_optical_pose = self.tf_buffer.lookup_transform(self.base_link, self.camera_rgb_optical_frame, t)
                depth_optical_pose = self.tf_buffer.lookup_transform(self.base_link, self.camera_depth_optical_frame, t)
                all_tf2_frames_as_string = self.tf_buffer.all_frames_as_string()
                self.tf2_dict = {}
                transform_strings = all_tf2_frames_as_string.split('\n')
                # get all of the other tf2 transforms
                # using the latest available frame as a fallback
                # if the current timestep frame isn't available
                for transform_string in transform_strings:
                    transform_tokens = transform_string.split(' ')
                    if len(transform_tokens) > 1:
                        k = transform_tokens[1]
                        try:

                            lookup_object = False
                            # Check for the detected object at the current time.
                            try:
                                k_pose = self.tf_buffer.lookup_transform(self.base_link, k, t)
                                lookup_object = True
                            except (tf2.ExtrapolationException, tf2.ConnectivityException) as e:
                                pass
                            if not lookup_object:
                                # If we can't get the current time for the object,
                                # get the latest available. This particular case will be common.
                                # This is because the object detection srcipt can only run when the
                                # arm is out of the way.
                                k_pose = self.tf_buffer.lookup_transform(self.base_link, k, latest_available_time_lookup)
                            k_pose = self.tf_buffer.lookup_transform(self.base_link, k, t)

                            k_xyz_qxqyqzqw = [
                                k_pose.transform.translation.x,
                                k_pose.transform.translation.y,
                                k_pose.transform.translation.z,
                                k_pose.transform.rotation.x,
                                k_pose.transform.rotation.y,
                                k_pose.transform.rotation.z,
                                k_pose.transform.rotation.w]
                            self.tf2_dict[k] = k_xyz_qxqyqzqw
                        except (tf2.ExtrapolationException, tf2.ConnectivityException) as e:
                            pass

                # don't load the yaml because it can take up to 0.2 seconds
                all_tf2_frames_as_yaml = self.tf_buffer.all_frames_as_yaml()
                self.tf2_json = json.dumps(self.tf2_dict)

                have_data = True
            except (tf2.LookupException, tf2.ExtrapolationException, tf2.ConnectivityException) as e:
                rospy.logwarn_throttle(
                    10.0,
                    'Collector transform lookup Failed: %s to %s, %s, %s'
                    ' at image time: %s and local time: %s '
                    '\nNote: This message may print >1000x less often than the problem occurs.' %
                    (self.base_link, self.camera_frame, self.ee_frame,
                     str(self.object), str(t), str(latest_available_time_lookup)))

                have_data = False
                attempts += 1
                # rospy.sleep(0.0)
                if attempts > max_attempts - backup_timestamp_attempts:
                    rospy.logwarn_throttle(
                        10.0,
                        'Collector failed to use the rgb image rosmsg timestamp, '
                        'trying latest available time as backup. '
                        'Note: This message may print >1000x less often than the problem occurs.')
                    # try the backup timestamp even though it will be less accurate
                    t = latest_available_time_lookup
                if attempts > max_attempts:
                    # Could not look up one of the transforms -- either could
                    # not look up camera, endpoint, or object.
                    raise e

        if t == latest_available_time_lookup:
            # Use either the latest available timestamp or
            # the local timestamp as backup,
            # even though it will be less accurate
            if self.rgb_time is not None:
                t = self.rgb_time
            else:
                t = local_time
        c_xyz_quat = pose_to_vec_quat_list(c_pose)
        rgb_optical_xyz_quat = pose_to_vec_quat_list(rgb_optical_pose)
        depth_optical_xyz_quat = pose_to_vec_quat_list(depth_optical_pose)
        ee_xyz_quat = pose_to_vec_quat_list(ee_pose)
        if self.object:
            obj_xyz_quat = pose_to_vec_quat_list(obj_pose)

        self.current_ee_pose = pm.fromTf(pose_to_vec_quat_pair(ee_pose))

        self.data["nsecs"].append(np.copy(self.t.nsecs)) # time
        self.data["secs"].append(np.copy(self.t.secs)) # time
        self.data["pose"].append(np.copy(ee_xyz_quat)) # end effector pose (6 DOF)
        self.data["camera"].append(np.copy(c_xyz_quat)) # camera pose (6 DOF)

        if self.object:
            self.data["object_pose"].append(np.copy(obj_xyz_quat))
        elif 'move_to_home' in label_to_check:
            self.data["object_pose"].append(self.home_xyz_quat)
            # TODO(ahundt) should object pose be all 0 or NaN when there is no object?
            # self.data["object_pose"].append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            raise ValueError("Attempted to log unsupported "
                             "object pose data for action_label " +
                             str(action_label))
        self.data["camera_rgb_optical_frame_pose"].append(rgb_optical_xyz_quat)
        self.data["camera_depth_optical_frame_pose"].append(depth_optical_xyz_quat)
        #plt.figure()
        #plt.imshow(self.rgb_img)
        #plt.show()
        # print("jpg size={}, png size={}".format(sys.getsizeof(img_jpeg), sys.getsizeof(depth_png)))
        self.data["image"].append(img_jpeg) # encoded as JPEG
        self.data["depth_image"].append(depth_png)
        self.data["gripper"].append(self.gripper_msg.gPO / 255.)

        # TODO(cpaxton): verify
        if not self.task.validLabel(action_label):
            raise RuntimeError("action not recognized: " + str(action_label))

        action = self.task.index(action_label)
        self.data["label"].append(action)  # integer code for high-level action

        # Take Mutex ---
        with self.mutex:
            self.data["q"].append(np.copy(self.q)) # joint position
            self.data["dq"].append(np.copy(self.dq)) # joint velocuity
            self.data["info"].append(np.copy(self.info))  # string description of current step
            self.data["rgb_info_D"].append(self.rgb_info.D)
            self.data["rgb_info_K"].append(self.rgb_info.K)
            self.data["rgb_info_R"].append(self.rgb_info.R)
            self.data["rgb_info_P"].append(self.rgb_info.P)
            self.data["rgb_info_distortion_model"].append(self.rgb_info.distortion_model)
            self.data["depth_info_D"].append(self.depth_info.D)
            self.data["depth_info_K"].append(self.depth_info.K)
            self.data["depth_info_R"].append(self.depth_info.R)
            self.data["depth_info_P"].append(self.depth_info.P)
            self.data["depth_distortion_model"].append(self.depth_info.distortion_model)
            if self.object:
                self.data["object"].append(np.copy(self.object))
            else:
                self.data["object"].append('none')

        self.data["all_tf2_frames_as_yaml"].append(all_tf2_frames_as_yaml)
        self.data["all_tf2_frames_from_base_link_vec_quat_xyzxyzw_json"].append(self.tf2_json)

        return True


def uint16_depth_image_to_png_numpy(cv_image):
    # split into two channels with a third zero channel
    rgb_np_image = encode_depth_numpy(cv_image)
    # plt.imshow(cv_image, cmap='nipy_spectral')
    # plt.imshow(rgb_np_image)
    # plt.pause(.01)
    # plt.draw()
    bytevalues = rgb_as_png_binary_bytes(rgb_np_image)
    return bytevalues

def rgb_as_png_binary_bytes(rgb_np_image):
    pil_image = PIL.Image.fromarray(rgb_np_image, mode='RGB')
    output = io.BytesIO()
    pil_image.save(output, format="PNG")
    bytevalues = output.getvalue()
    return bytevalues

def encode_depth_numpy(cv_image, order='bgr'):
    # split into two channels with a third zero channel
    r = np.array(np.divide(cv_image, 256), dtype=np.uint8)
    g = np.array(np.mod(cv_image, 256), dtype=np.uint8)
    b = np.zeros(cv_image.shape, dtype=np.uint8)
    # If we are using opencv to encode we want to use bgr for
    # the encoding step to ensure the color order is correct
    # when it is decoded
    if order == 'rgb':
        rgb_np_image = np.stack([r, g, b], axis=-1)
    elif order == 'bgr':
        rgb_np_image = np.stack([b, g, r], axis=-1)
    else:
        raise ValueError('encode_depth_numpy unsupported encoding' + str(order))

    # rospy.loginfo('rgb_np_image shape: ' + str(rgb_np_image.shape) + ' depth sequence number: ' + str(msg.header.seq))
    # plt.imshow(cv_image, cmap='nipy_spectral')
    # plt.imshow(rgb_np_image)
    # plt.pause(.01)
    # plt.draw()
    return rgb_np_image


if __name__ == '__main__':
    pass


