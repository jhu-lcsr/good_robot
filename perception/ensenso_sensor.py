"""
Interface to the Ensenso N* Sensor
Author: Jeff Mahler
"""
import IPython
import logging
import numpy as np
import os
import struct
import sys
import time

try:
    from cv_bridge import CvBridge, CvBridgeError
    import rospy
    from sensor_msgs.msg import CameraInfo, PointCloud2
    import sensor_msgs.point_cloud2 as pc2
except ImportError:
    logging.warning("Failed to import ROS in ensenso_sensor.py. ROS functionality not available")
    
from .constants import MM_TO_METERS, INTR_EXTENSION
from . import CameraIntrinsics, CameraSensor, ColorImage, DepthImage, Image

class EnsensoSensor(CameraSensor):
    """ Class for interfacing with an Ensenso N* sensor.
    """
    def __init__(self, frame='ensenso'):
        # set member vars
        self._frame = frame
        self._initialized = False
        self._format = None
        self._camera_intr = None
        self._cur_depth_im = None
        self._running = False
        
    def __del__(self):
        """Automatically stop the sensor for safety.
        """
        if self.is_running:
            self.stop()
        
    def _set_format(self, msg):
        """ Set the buffer formatting. """
        num_points = msg.height * msg.width
        self._format = '<' + num_points * 'ffff'
            
    def _set_camera_properties(self, msg):
        """ Set the camera intrinsics from an info msg. """
        focal_x = msg.K[0]
        focal_y = msg.K[4]
        center_x = msg.K[2]
        center_y = msg.K[5]
        im_height = msg.height
        im_width = msg.width
        self._camera_intr = CameraIntrinsics(self._frame, focal_x, focal_y,
                                             center_x, center_y,
                                             height=im_height,
                                             width=im_width)

    def _depth_im_from_pointcloud(self, msg):
        """ Convert a pointcloud2 message to a depth image. """
        # set format
        if self._format is None:
            self._set_format(msg)

        # rescale camera intr in case binning is turned on
        if msg.height != self._camera_intr.height:
            rescale_factor = float(msg.height) / self._camera_intr.height
            self._camera_intr = self._camera_intr.resize(rescale_factor)
            
        # read num points
        num_points = msg.height * msg.width
            
        # read buffer
        raw_tup = struct.Struct(self._format).unpack_from(msg.data, 0)
        raw_arr = np.array(raw_tup)

        # subsample depth values and reshape
        depth_ind = 2 + 4 * np.arange(num_points)
        depth_buf = raw_arr[depth_ind]
        depth_arr = depth_buf.reshape(msg.height, msg.width)
        depth_im = DepthImage(depth_arr, frame=self._frame)

        return depth_im

    def _pointcloud_callback(self, msg):
        """ Callback for handling point clouds. """
        self._cur_depth_im = self._depth_im_from_pointcloud(msg)
        
    def _camera_info_callback(self, msg):
        """ Callback for reading camera info. """
        self._camera_info_sub.unregister()
        self._set_camera_properties(msg)
        
    @property
    def ir_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the Ensenso IR camera.
        """
        return self._camera_intr

    @property
    def is_running(self):
        """bool : True if the stream is running, or false otherwise.
        """
        return self._running

    @property
    def frame(self):
        """:obj:`str` : The reference frame of the sensor.
        """
        return self._frame

    def start(self):
        """ Start the sensor """
        # initialize subscribers
        self._pointcloud_sub = rospy.Subscriber('/%s/depth/points' %(self.frame), PointCloud2, self._pointcloud_callback)
        self._camera_info_sub = rospy.Subscriber('/%s/left/camera_info' %(self.frame), CameraInfo, self._camera_info_callback)

        while self._camera_intr is None:
            time.sleep(0.1)
        
        self._running = True

    def stop(self):
        """ Stop the sensor """
        # check that everything is running
        if not self._running:
            logging.warning('Ensenso not running. Aborting stop')
            return False

        # stop subs
        self._pointcloud_sub.unregister()
        self._camera_info_sub.unregister()
        self._running = False
        return True

    def frames(self):
        """Retrieve a new frame from the Ensenso and convert it to a ColorImage,
        a DepthImage, and an IrImage.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`, :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the current frame.

        Raises
        ------
        RuntimeError
            If the Ensenso stream is not running.
        """
        # wait for a new image
        while self._cur_depth_im is None:
            time.sleep(0.01)
            
        # read next image
        depth_im = self._cur_depth_im
        color_im = ColorImage(np.zeros([depth_im.height,
                                        depth_im.width,
                                        3]).astype(np.uint8), frame=self._frame)
        self._cur_depth_im = None
        return color_im, depth_im, None

    def median_depth_img(self, num_img=1, fill_depth=0.0):
        """Collect a series of depth images and return the median of the set.

        Parameters
        ----------
        num_img : int
            The number of consecutive frames to process.

        Returns
        -------
        :obj:`DepthImage`
            The median DepthImage collected from the frames.
        """
        depths = []

        for _ in range(num_img):
            _, depth, _ = self.frames()
            depths.append(depth)

        median_depth = Image.median_images(depths)
        median_depth.data[median_depth.data == 0.0] = fill_depth
        return median_depth
        
