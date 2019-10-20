#!/usr/bin/env python
"""OpenCV feature detectors with ros CompressedImage Topics in python.

This example subscribes to a ros topic containing sensor_msgs 
CompressedImage. It converts the CompressedImage into a numpy.ndarray, 
then detects and marks features in that image. It finally displays 
and publishes the new image - again as CompressedImage topic.

Based on code at http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber by Simon Haller <simon.haller at uibk.ac.at>
"""
__author__ =  'Andrew Hundt <ATHundt@gmail.com>'
__version__=  '0.1'
__license__ = 'BSD'
# Python libs
import sys, time

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from sensor_msgs.msg import CameraInfo
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from threading import Lock
import message_filters

VERBOSE=False

class ros_camera:

    def __init__(self, synchronize=False):

        # http://wiki.ros.org/depth_image_proc
        # http://www.ros.org/reps/rep-0118.html
        # http://wiki.ros.org/rgbd_launch
        # we will be getting 16 bit integer values in milimeters
        self.rgb_topic = "/camera/rgb/image_rect_color"
        # raw means it is in the format provided by the openi drivers, 16 bit int
        self.depth_topic = "/camera/depth_registered/hw_registered/image_rect"
        self.camera_depth_info_topic = "/camera/rgb/camera_info"
        self.camera_rgb_info_topic = "/camera/depth_registered/camera_info"
        '''Initialize ros publisher, ros subscriber'''
        # # topic where we publish
        # self.image_pub = rospy.Publisher("/output/image_raw/compressed",
        #     CompressedImage)
        # # self.bridge = CvBridge()

        # # subscribed Topic
        # self.subscriber = rospy.Subscriber("/camera/image/compressed",
        #     CompressedImage, self.callback,  queue_size = 1)
        self.depth_img = None
        self.rgb_img = None
        self.rgb_time = None
        self.mutex = Lock()
        self._bridge = CvBridge()
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

                with self.mutex:
                    self.rgb_time = msg.header.stamp
                    self.rgb_img = rgb_img
                    # self.depth_info = depth_camera_info
                    # self.rgb_info = rgb_camera_info
                    self.depth_img_time = msg.header.stamp
                    # self.depth_img = np_image
                    # self.depth_img = img_str
                    self.depth_img_as_rgb = bytevalues
                    self.depth_img = cv_depth_image
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
                # rgb_img = cv2.imencode('.jpg', cv_image)[1].tobytes()
                # rgb_img = GetJpeg(np.asarray(cv_image))

                with self.mutex:
                    self.rgb_time = msg.header.stamp
                    self.rgb_img = cv_image
                    # self.rgb_img = rgb_img
                # print('_rgbCb()')
            #print(self.rgb_img)
        except CvBridgeError as e:
            rospy.logwarn(str(e))

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
                # depth_encoded_as_rgb_numpy = encode_depth_numpy(cv_image)
                # bytevalues = cv2.imencode('.png', depth_encoded_as_rgb_numpy)[1].tobytes()

                with self.mutex:
                    self.depth_img_time = msg.header.stamp
                    # self.depth_img = np_image
                    # self.depth_img = img_str
                    # self.depth_img = bytevalues
                    # self.depth_img = depth_encoded_as_rgb_numpy
                    self.depth_img = cv_image
                # print('_depthCb()')
                # print (self.depth_img)
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

    def frames(self):
        rgb_image = None
        depth_image = None
        with self.mutex:
            if self.rgb_img is not None:
                rgb_image = self.rgb_img.copy()
            if self.depth_img is not None:
                depth_image = self.depth_img.copy()
        return rgb_image, depth_image, None

def main(args):
    '''Initializes and cleanup ros node'''
    print('Before you run test_ros_images.py, make sure you have first started the ROS node for reading the primesense sensor data:'
          ' roslaunch openni2_launch openni2.launch depth_registration:=true')
    ic = ros_camera()
    rospy.init_node('ros_camera', anonymous=True)
    try:
        time.sleep(1)
        i = 0
        while True:
            print(i)
            # rospy.spin_one()
            rgb, depth, _ = ic.frames()
            if rgb is not None:
                cv2.imshow('color.png', rgb)
            if depth is not None:
                # Multiply just so you can see the values, don't do that with real data
                depth *= 100
                cv2.imshow('depth.png', depth)
            cv2.waitKey(1)
            i += 1

    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)