import time
import logging

import numpy as np

try:
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image as ImageMessage
    from std_srvs.srv import Empty
    from perception.srv import ConnectCamera, GetDeviceList, GetFrame, TriggerImage
except ImportError:
    logging.warning("Failed to import ROS in phoxi_sensor.py. PhoXiSensor functionality unavailable.")

from . import CameraSensor, DepthImage, ColorImage, GrayscaleImage, CameraIntrinsics, Image, SensorUnresponsiveException

class PhoXiSensor(CameraSensor):
    """Class for interfacing with a PhoXi Structured Light Sensor.
    """

    def __init__(self, frame='phoxi', device_name='2018-02-020-LC3', size='small'):
        """Initialize a PhoXi Sensor.

        Parameters
        ----------
        frame : str
            A name for the frame in which depth images, normal maps, and RGB images are returned.
        device_name : str
            The string name of the PhoXi device (SN listed on sticker on back sensor).
            Old PhoXi: 1703005
            New PhoXi: 2018-02-020-LC3
        size : str
            An indicator for which size of image is desired.
            Either 'large' (2064x1544) or 'small' (1032x772).
        """

        self._frame = frame
        self._device_name = str(device_name)
        self._camera_intr = None
        self._running = False
        self._bridge = CvBridge()

        self._cur_color_im = None
        self._cur_depth_im = None
        self._cur_normal_map = None

        # Set up camera intrinsics for the sensor
        width, height = 2064, 1544
        focal_x, focal_y = 2244., 2244.
        center_x, center_y = 1023., 768.
        if size == 'small':
            width = 1032
            height = 772
            focal_x = focal_x / 2
            focal_y = focal_y / 2
            center_x = center_x / 2
            center_y = center_y / 2

            if str(device_name) == '1703005':
                focal_x = focal_y = 1105.

        self._camera_intr = CameraIntrinsics(self._frame, focal_x, focal_y,
                                             center_x, center_y,
                                             height=height, width=width)

    def __del__(self):
        """Automatically stop the sensor for safety.
        """
        if self.is_running:
            self.stop()

    @property
    def color_intrinsics(self):
        """CameraIntrinsics : The camera intrinsics for the PhoXi Greyscale camera.
        """
        return self._camera_intr

    @property
    def ir_intrinsics(self):
        """CameraIntrinsics : The camera intrinsics for the PhoXi IR camera.
        """
        return self._camera_intr

    @property
    def is_running(self):
        """bool : True if the stream is running, or false otherwise.
        """
        return self._running

    @property
    def frame(self):
        """str : The reference frame of the sensor.
        """
        return self._frame

    @property
    def color_frame(self):
        """str : The reference frame of the sensor.
        """
        return self._frame

    @property
    def ir_frame(self):
        """str : The reference frame of the sensor.
        """
        return self._frame

    def start(self):
        """Start the sensor.
        """
        if rospy.get_name() == '/unnamed':
            raise ValueError('PhoXi sensor must be run inside a ros node!')

        # Connect to the cameras
        if not self._connect_to_sensor():
            self._running = False
            return False

        # Set up subscribers for camera data
        self._color_im_sub = rospy.Subscriber('/phoxi_camera/texture', ImageMessage, self._color_im_callback)
        self._depth_im_sub = rospy.Subscriber('/phoxi_camera/depth_map', ImageMessage, self._depth_im_callback)
        self._normal_map_sub = rospy.Subscriber('/phoxi_camera/normal_map', ImageMessage, self._normal_map_callback)

        self._running = True

        return True

    def stop(self):
        """Stop the sensor.
        """
        # Check that everything is running
        if not self._running:
            logging.warning('PhoXi not running. Aborting stop')
            return False

        # Stop the subscribers
        self._color_im_sub.unregister()
        self._depth_im_sub.unregister()
        self._normal_map_sub.unregister()

        # Disconnect from the camera
        rospy.ServiceProxy('phoxi_camera/disconnect_camera', Empty)()

        self._running = False

        return True

    def frames(self):
        """Retrieve a new frame from the PhoXi and convert it to a ColorImage,
        a DepthImage, and an IrImage.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`, :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the current frame.
        """
        # Run a software trigger
        times = []

        rospy.ServiceProxy('phoxi_camera/start_acquisition', Empty)()
        rospy.ServiceProxy('phoxi_camera/trigger_image', TriggerImage)()

        self._cur_color_im = None
        self._cur_depth_im = None
        self._cur_normal_map = None

        rospy.ServiceProxy('phoxi_camera/get_frame', GetFrame)(-1)

        max_time = 5.0
        time_waiting = 0.0
        while self._cur_color_im is None or self._cur_depth_im is None or self._cur_normal_map is None:
            time.sleep(0.05)
            time_waiting += 0.05
            if time_waiting > max_time:
                raise SensorUnresponsiveException('PhoXi sensor seems to be non-responsive')
        return self._cur_color_im, self._cur_depth_im, None

    def median_depth_img(self, num_img=1, fill_depth=0.0):
        """Collect a series of depth images and return the median of the set.

        Parameters
        ----------
        num_img : int
            The number of consecutive frames to process.

        Returns
        -------
        DepthImage
            The median DepthImage collected from the frames.
        """
        depths = []

        for _ in range(num_img):
            _, depth, _ = self.frames()
            depths.append(depth)

        median_depth = Image.median_images(depths)
        median_depth.data[median_depth.data == 0.0] = fill_depth
        return median_depth

    def _connect_to_sensor(self):
        """Connect to the sensor.
        """
        name = self._device_name
        try:
            # Check if device is actively in list
            rospy.wait_for_service('phoxi_camera/get_device_list')
            device_list = rospy.ServiceProxy('phoxi_camera/get_device_list', GetDeviceList)().out
            if not str(name) in device_list:
                logging.error('PhoXi sensor {} not in list of active devices'.format(name))
                return False

            success = rospy.ServiceProxy('phoxi_camera/connect_camera', ConnectCamera)(name).success
            if not success:
                logging.error('Could not connect to PhoXi sensor {}'.format(name))
                return False

            logging.debug('Connected to PhoXi Sensor {}'.format(name))
            return True

        except rospy.ServiceException as e:
            logging.error('Service call failed: {}'.format(e))
            return False

    def _color_im_callback(self, msg):
        """Callback for handling textures (greyscale images).
        """
        try:
            data = self._bridge.imgmsg_to_cv2(msg)
            if np.max(data) > 255.0:
                data = 255.0 * data / 1200.0 # Experimentally set value for white
            data = np.clip(data, 0., 255.0).astype(np.uint8)
            gsimage = GrayscaleImage(data, frame=self._frame)
            self._cur_color_im = gsimage.to_color()
        except:
            self._cur_color_im = None

    def _depth_im_callback(self, msg):
        """Callback for handling depth images.
        """
        try:
            self._cur_depth_im = DepthImage(self._bridge.imgmsg_to_cv2(msg) / 1000.0, frame=self._frame)
        except:
            self._cur_depth_im = None

    def _normal_map_callback(self, msg):
        """Callback for handling normal maps.
        """
        try:
            self._cur_normal_map = self._bridge.imgmsg_to_cv2(msg)
        except:
            self._cur_normal_map = None
