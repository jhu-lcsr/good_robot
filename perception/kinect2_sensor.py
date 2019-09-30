"""
Class for interfacing with the Primesense RGBD sensor
Author: Jeff Mahler
"""
import copy
import logging
import numpy as np
import os
import time

try:
    import cv2
    import pylibfreenect2 as lf2
except:
    logging.warning('Unable to import pylibfreenect2. Python-only Kinect driver may not work properly.')

try:
    from cv_bridge import CvBridge, CvBridgeError
    import rospy
    import sensor_msgs.msg
    import sensor_msgs.point_cloud2 as pc2
except ImportError:
    logging.warning("Failed to import ROS in Kinect2_sensor.py. Kinect will not be able to be used in bridged mode")
    

from .constants import MM_TO_METERS, INTR_EXTENSION
from .camera_intrinsics import CameraIntrinsics
from .camera_sensor import CameraSensor
from .image import ColorImage, DepthImage, IrImage, Image

class Kinect2PacketPipelineMode:
    """Type of pipeline for Kinect packet processing.
    """
    OPENGL = 0
    CPU = 1

class Kinect2FrameMode:
    """Type of frames that Kinect processes.
    """
    COLOR_DEPTH = 0
    COLOR_DEPTH_IR = 1

class Kinect2RegistrationMode:
    """Kinect registration mode.
    """
    NONE = 0
    COLOR_TO_DEPTH = 1

class Kinect2DepthMode:
    """Kinect depth mode setting.
    """
    METERS = 0
    MILLIMETERS = 1

class Kinect2BridgedQuality:
    """Kinect quality for bridged mode
    """
    HD = "hd"
    QUARTER_HD = "qhd"
    SD = "sd"

class Kinect2Sensor(CameraSensor):
    # constants for image height and width (in case they're needed somewhere)
    """Class for interacting with a Kinect v2 RGBD sensor directly through protonect driver.
    https://github.com/OpenKinect/libfreenect2
    """

    #Constants for image height and width (in case they're needed somewhere)
    COLOR_IM_HEIGHT = 1080
    COLOR_IM_WIDTH = 1920
    DEPTH_IM_HEIGHT = 424
    DEPTH_IM_WIDTH = 512

    def __init__(self, packet_pipeline_mode = Kinect2PacketPipelineMode.CPU,
                 registration_mode = Kinect2RegistrationMode.COLOR_TO_DEPTH,
                 depth_mode = Kinect2DepthMode.METERS,
                 device_num=0, frame=None):
        """Initialize a Kinect v2 sensor directly to the protonect driver with the given configuration.
        When kinect is connected to the protonect driver directly, the iai_kinect kinect_bridge cannot be run at the same time
        Parameters
        ----------
        packet_pipeline_mode : int
            Either Kinect2PacketPipelineMode.OPENGL or
            Kinect2PacketPipelineMode.CPU -- indicates packet processing type.

        registration_mode : int
            Either Kinect2RegistrationMode.NONE or
            Kinect2RegistrationMode.COLOR_TO_DEPT -- The mode for registering
            a color image to the IR camera frame of reference.

        depth_mode : int
            Either Kinect2DepthMode.METERS or Kinect2DepthMode.MILLIMETERS --
            the units for depths returned from the Kinect frame arrays.

        device_num : int
            The sensor's device number on the USB bus.

        frame : :obj:`str`
            The name of the frame of reference in which the sensor resides.
            If None, this will be set to 'kinect2_num', where num is replaced
            with the device number.
       """
        self._device = None
        self._running = False
        self._packet_pipeline_mode = packet_pipeline_mode
        self._registration_mode = registration_mode
        self._depth_mode = depth_mode
        self._device_num = device_num
        self._frame = frame

        if self._frame is None:
            self._frame = 'kinect2_%d' %(self._device_num)
        self._color_frame = '%s_color' %(self._frame)
        self._ir_frame = self._frame # same as color since we normally use this one

    def __del__(self):
        """Automatically stop the sensor for safety.
        """
        if self.is_running:
            self.stop()

    @property
    def color_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the Kinect's color camera.
        """
        if self._device is None:
            raise RuntimeError('Kinect2 device %s not runnning. Cannot return color intrinsics')
        camera_params = self._device.getColorCameraParams()
        return CameraIntrinsics(self._color_frame, camera_params.fx, camera_params.fy,
                                camera_params.cx, camera_params.cy)

    @property
    def ir_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the Kinect's IR camera.
        """
        if self._device is None:
            raise RuntimeError('Kinect2 device %s not runnning. Cannot return IR intrinsics')
        camera_params = self._device.getIrCameraParams()
        return CameraIntrinsics(self._ir_frame, camera_params.fx, camera_params.fy,
                                camera_params.cx, camera_params.cy,
                                height=Kinect2Sensor.DEPTH_IM_HEIGHT,
                                width=Kinect2Sensor.DEPTH_IM_WIDTH)

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

    @property
    def color_frame(self):
        """:obj:`str` : The reference frame of the color sensor.
        """
        return self._color_frame

    @property
    def ir_frame(self):
        """:obj:`str` : The reference frame of the IR sensor.
        """
        return self._ir_frame

    def start(self):
        """Starts the Kinect v2 sensor stream.

        Raises
        ------
        IOError
            If the Kinect v2 is not detected.
        """
        # open packet pipeline
        if self._packet_pipeline_mode == Kinect2PacketPipelineMode.OPENGL:
            self._pipeline = lf2.OpenGLPacketPipeline()
        elif self._packet_pipeline_mode == Kinect2PacketPipelineMode.CPU:
            self._pipeline = lf2.CpuPacketPipeline()

        # setup logger
        self._logger = lf2.createConsoleLogger(lf2.LoggerLevel.Warning)
        lf2.setGlobalLogger(self._logger)

        # check devices
        self._fn_handle = lf2.Freenect2()
        self._num_devices = self._fn_handle.enumerateDevices()
        if self._num_devices == 0:
            raise IOError('Failed to start stream. No Kinect2 devices available!')
        if self._num_devices <= self._device_num:
            raise IOError('Failed to start stream. Device num %d unavailable!' %(self._device_num))

        # open device
        self._serial = self._fn_handle.getDeviceSerialNumber(self._device_num)
        self._device = self._fn_handle.openDevice(self._serial, pipeline=self._pipeline)

        # add device sync modes
        self._listener = lf2.SyncMultiFrameListener(
            lf2.FrameType.Color | lf2.FrameType.Ir | lf2.FrameType.Depth)
        self._device.setColorFrameListener(self._listener)
        self._device.setIrAndDepthFrameListener(self._listener)

        # start device
        self._device.start()

        # open registration
        self._registration = None
        if self._registration_mode == Kinect2RegistrationMode.COLOR_TO_DEPTH:
            logging.debug('Using color to depth registration')
            self._registration = lf2.Registration(self._device.getIrCameraParams(),
                                                  self._device.getColorCameraParams())
        self._running = True

    def stop(self):
        """Stops the Kinect2 sensor stream.

        Returns
        -------
        bool
            True if the stream was stopped, False if the device was already
            stopped or was not otherwise available.
        """
        # check that everything is running
        if not self._running or self._device is None:
            logging.warning('Kinect2 device %d not runnning. Aborting stop' %(self._device_num))
            return False

        # stop the device
        self._device.stop()
        self._device.close()
        self._device = None
        self._running = False
        return True

    def frames(self, skip_registration=False):
        """Retrieve a new frame from the Kinect and convert it to a ColorImage,
        a DepthImage, and an IrImage.

        Parameters
        ----------
        skip_registration : bool
            If True, the registration step is skipped.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`, :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the current frame.

        Raises
        ------
        RuntimeError
            If the Kinect stream is not running.
        """
        color_im, depth_im, ir_im, _ = self._frames_and_index_map(skip_registration=skip_registration)
        return color_im, depth_im, ir_im

    def median_depth_img(self, num_img=1):
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

        return Image.median_images(depths)

    def _frames_and_index_map(self, skip_registration=False):
        """Retrieve a new frame from the Kinect and return a ColorImage,
        DepthImage, IrImage, and a map from depth pixels to color pixel indices.

        Parameters
        ----------
        skip_registration : bool
            If True, the registration step is skipped.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`, :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the current frame, and an
            ndarray that maps pixels of the depth image to the index of the
            corresponding pixel in the color image.

        Raises
        ------
        RuntimeError
            If the Kinect stream is not running.
        """
        if not self._running:
            raise RuntimeError('Kinect2 device %s not runnning. Cannot read frames' %(self._device_num))

        # read frames
        frames = self._listener.waitForNewFrame()
        unregistered_color = frames['color']
        distorted_depth = frames['depth']
        ir = frames['ir']

        # apply color to depth registration
        color_frame = self._color_frame
        color = unregistered_color
        depth = distorted_depth
        color_depth_map = np.zeros([depth.height, depth.width]).astype(np.int32).ravel()
        if not skip_registration and self._registration_mode == Kinect2RegistrationMode.COLOR_TO_DEPTH:
            color_frame = self._ir_frame
            depth = lf2.Frame(depth.width, depth.height, 4, lf2.FrameType.Depth)
            color = lf2.Frame(depth.width, depth.height, 4, lf2.FrameType.Color)
            self._registration.apply(unregistered_color, distorted_depth, depth, color, color_depth_map=color_depth_map)

        # convert to array (copy needed to prevent reference of deleted data
        color_arr = copy.copy(color.asarray())
        color_arr[:,:,[0,2]] = color_arr[:,:,[2,0]] # convert BGR to RGB
        color_arr[:,:,0] = np.fliplr(color_arr[:,:,0])
        color_arr[:,:,1] = np.fliplr(color_arr[:,:,1])
        color_arr[:,:,2] = np.fliplr(color_arr[:,:,2])
        color_arr[:,:,3] = np.fliplr(color_arr[:,:,3])
        depth_arr = np.fliplr(copy.copy(depth.asarray()))
        ir_arr = np.fliplr(copy.copy(ir.asarray()))

        # convert meters
        if self._depth_mode == Kinect2DepthMode.METERS:
            depth_arr = depth_arr * MM_TO_METERS

        # Release and return
        self._listener.release(frames)
        return (ColorImage(color_arr[:,:,:3], color_frame),
                DepthImage(depth_arr, self._ir_frame),
                IrImage(ir_arr.astype(np.uint16), self._ir_frame),
                color_depth_map)

class KinectSensorBridged(CameraSensor):
    """Class for interacting with a Kinect v2 RGBD sensor through the kinect bridge
    https://github.com/code-iai/iai_kinect2. This is preferrable for visualization and debug
    because the kinect bridge will continuously publish image and point cloud info.
    """

    def __init__(self, quality=Kinect2BridgedQuality.HD, frame='kinect2_rgb_optical_frame'):
        """Initialize a Kinect v2 sensor which connects to the iai_kinect2 bridge
        ----------
        quality : :obj:`str`
            The quality (HD, Quarter-HD, SD) of the image data that should be subscribed to
        frame : :obj:`str`
            The name of the frame of reference in which the sensor resides.
            If None, this will be set to 'kinect2_rgb_optical_frame'
       """
        # set member vars
        self._frame = frame

        self.topic_image_color = '/kinect2/%s/image_color_rect' %(quality)
        self.topic_image_depth = '/kinect2/%s/image_depth_rect' %(quality)
        self.topic_info_camera = '/kinect2/%s/camera_info' %(quality)
        
        self._initialized = False
        self._format = None
        self._camera_intr = None
        self._cur_depth_im = None
        self._running = False
        self._bridge = CvBridge()
        
    def __del__(self):
        """Automatically stop the sensor for safety.
        """
        if self.is_running:
            self.stop()
            
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

    def _process_image_msg(self, msg):
        """ Process an image message and return a numpy array with the image data
        Returns
        -------
        :obj:`numpy.ndarray` containing the image in the image message

        Raises
        ------
        CvBridgeError
            If the bridge is not able to convert the image
        """
        encoding = msg.encoding
        try:
            image = self._bridge.imgmsg_to_cv2(msg, encoding)
        except CvBridgeError as e:
            rospy.logerr(e)
        return image
        
    def _color_image_callback(self, image_msg):
        """ subscribe to image topic and keep it up to date
        """
        color_arr = self._process_image_msg(image_msg)
        self._cur_color_im = ColorImage(color_arr[:,:,::-1], self._frame)
 
    def _depth_image_callback(self, image_msg):
        """ subscribe to depth image topic and keep it up to date
        """
        encoding = image_msg.encoding
        try:
            depth_arr = self._bridge.imgmsg_to_cv2(image_msg, encoding)
            import pdb; pdb.set_trace()

        except CvBridgeError as e:
            rospy.logerr(e)
        depth = np.array(depth_arr*MM_TO_METERS, np.float32)
        self._cur_depth_im = DepthImage(depth, self._frame)

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
        self._image_sub = rospy.Subscriber(self.topic_image_color, sensor_msgs.msg.Image, self._color_image_callback)
        self._depth_sub = rospy.Subscriber(self.topic_image_depth, sensor_msgs.msg.Image, self._depth_image_callback)
        self._camera_info_sub = rospy.Subscriber(self.topic_info_camera, sensor_msgs.msg.CameraInfo, self._camera_info_callback)
        
        timeout = 10
        try:
            rospy.loginfo("waiting to recieve a message from the Kinect")
            rospy.wait_for_message(self.topic_image_color, sensor_msgs.msg.Image, timeout=timeout)
            rospy.wait_for_message(self.topic_image_depth, sensor_msgs.msg.Image, timeout=timeout)
            rospy.wait_for_message(self.topic_info_camera, sensor_msgs.msg.CameraInfo, timeout=timeout)
        except rospy.ROSException as e:
            print("KINECT NOT FOUND")
            rospy.logerr("Kinect topic not found, Kinect not started")
            rospy.logerr(e)

        while self._camera_intr is None:
            time.sleep(0.1)
        
        self._running = True

    def stop(self):
        """ Stop the sensor """
        # check that everything is running
        if not self._running:
            logging.warning('Kinect not running. Aborting stop')
            return False

        # stop subs
        self._image_sub.unregister()
        self._depth_sub.unregister()
        self._camera_info_sub.unregister

        self._running = False
        return True

    def frames(self):
        """Retrieve a new frame from the Ensenso and convert it to a ColorImage,
        a DepthImage, IrImage is always none for this type

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`, :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the current frame.

        Raises
        ------
        RuntimeError
            If the Kinect stream is not running.
        """
        # wait for a new image
        while self._cur_depth_im is None or self._cur_color_im is None:
            time.sleep(0.01)
            
        # read next image
        depth_im = self._cur_depth_im
        color_im = self._cur_color_im

        self._cur_color_im = None
        self._cur_depth_im = None

        #TODO add ir image
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


class VirtualKinect2Sensor(CameraSensor):
    """Class for a virtual Kinect v2 sensor that uses pre-captured images
    stored to disk instead of actually connecting to a sensor.
    For debugging purposes.
    """ 

    def __init__(self, path_to_images, frame=None):
        """Create a new virtualized Kinect v2 sensor.

        This requires a directory containing a specific set of files.

        First, the directory must contain a set of images, where each
        image has three files:
        - color_{#}.png
        - depth_{#}.npy
        - ir_{#}.npy
        In these, the {#} is replaced with the integer index for the
        image. These indices should start at zero and increase
        consecutively.

        Second, the directory must contain CameraIntrisnics files
        for the color and ir cameras:
        - {frame}_color.intr
        - {frame}_ir.intr
        In these, the {frame} is replaced with the reference frame
        name that is passed as a parameter to this function.

        Parameters
        ----------
        path_to_images : :obj:`str`
            The path to a directory containing images that the virtualized
            sensor will return.

        frame : :obj:`str`
            The name of the frame of reference in which the sensor resides.
            If None, this will be discovered from the files in the directory.
        """
        self._running = False
        self._path_to_images = path_to_images
        self._im_index = 0
        self._num_images = 0
        self._frame = frame
        filenames = os.listdir(self._path_to_images)

        # get number of images
        for filename in filenames:
            if filename.find('depth') != -1 and filename.endswith('.npy'):
                self._num_images += 1

        # set the frame dynamically
        if self._frame is None:
            for filename in filenames:
                file_root, file_ext = os.path.splitext(filename)
                color_ind = file_root.rfind('color')

                if file_ext == INTR_EXTENSION and color_ind != -1:
                    self._frame = file_root[:color_ind-1]
                    self._color_frame = file_root
                    self._ir_frame = file_root
                    break

        # load color intrinsics
        color_intr_filename = os.path.join(self._path_to_images, '%s_color.intr' %(self._frame))
        self._color_intr = CameraIntrinsics.load(color_intr_filename)
        ir_intr_filename = os.path.join(self._path_to_images, '%s_ir.intr' %(self._frame))
        self._ir_intr = CameraIntrinsics.load(ir_intr_filename)

    @property
    def path_to_images(self):
        """:obj:`str` : The path to a directory containing images that the virtualized
        sensor will return.
        """
        return self._path_to_images

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

    @property
    def color_frame(self):
        """:obj:`str` : The reference frame of the color sensor.
        """
        return self._color_frame

    @property
    def ir_frame(self):
        """:obj:`str` : The reference frame of the IR sensor.
        """
        return self._ir_frame

    @property
    def color_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the Kinect's color camera.
        """
        return self._color_intr

    @property
    def ir_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the Kinect's IR camera.
        """
        return self._ir_intr


    def start(self):
        """Starts the Kinect v2 sensor stream.

        In this virtualized sensor, this simply resets the image index to zero.
        Everytime start is called, we start the stream again at the first image.
        """
        self._im_index = 0
        self._running = True

    def stop(self):
        """Stops the Kinect2 sensor stream.

        Returns
        -------
        bool
            True if the stream was stopped, False if the device was already
            stopped or was not otherwise available.
        """
        if not self._running:
            return false
        self._running = False
        return True

    def frames(self):
        """Retrieve the next frame from the image directory and convert it to a ColorImage,
        a DepthImage, and an IrImage.

        Parameters
        ----------
        skip_registration : bool
            If True, the registration step is skipped.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`, :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the current frame.

        Raises
        ------
        RuntimeError
            If the Kinect stream is not running or if all images in the
            directory have been used.
        """
        if not self._running:
            raise RuntimeError('VirtualKinect2 device pointing to %s not runnning. Cannot read frames' %(self._path_to_images))

        if self._im_index > self._num_images:
            raise RuntimeError('VirtualKinect2 device is out of images')

        # read images
        color_filename = os.path.join(self._path_to_images, 'color_%d.png' %(self._im_index))
        color_im = ColorImage.open(color_filename, frame=self._frame)
        depth_filename = os.path.join(self._path_to_images, 'depth_%d.npy' %(self._im_index))
        depth_im = DepthImage.open(depth_filename, frame=self._frame)
        ir_filename = os.path.join(self._path_to_images, 'ir_%d.npy' %(self._im_index))
        ir_im = None
        if os.path.exists(ir_filename):
            ir_im = IrImage.open(ir_filename, frame=self._frame)
        self._im_index += 1
        return color_im, depth_im, ir_im

    def median_depth_img(self, num_img=1):
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

        return Image.median_images(depths)

class Kinect2SensorFactory:
    """ Factory class for Kinect2 sensors. """

    @staticmethod
    def sensor(sensor_type, cfg):
        """ Creates a Kinect2 sensor of the specified type.

        Parameters
        ----------
        sensor_type : :obj:`str`
            the type of the sensor (real or virtual)
        cfg : :obj:`YamlConfig`
            dictionary of parameters for sensor initialization
        """
        sensor_type = sensor_type.lower()
        if sensor_type == 'real':
            s = Kinect2Sensor(packet_pipeline_mode=cfg['pipeline_mode'],
                              device_num=cfg['device_num'],
                              frame=cfg['frame'])
        elif sensor_type == 'virtual':
            s = VirtualKinect2Sensor(cfg['image_dir'],
                                     frame=cfg['frame'])
        elif sensor_type == 'bridged':
            s = KinectSensorBridged(quality=cfg['quality'], frame=cfg['frame'])
        else:
            raise ValueError('Kinect2 sensor type %s not supported' %(sensor_type))
        return s

def load_images(cfg):
    """Helper function for loading a set of color images, depth images, and IR
    camera intrinsics.

    The config dictionary must have these keys:
        - prestored_data -- If 1, use the virtual sensor, else use a real sensor.
        - prestored_data_dir -- A path to the prestored data dir for a virtual sensor.
        - sensor/frame -- The frame of reference for the sensor.
        - sensor/device_num -- The device number for the real Kinect.
        - sensor/pipeline_mode -- The mode for the real Kinect's packet pipeline.
        - num_images -- The number of images to generate.

    Parameters
    ----------
    cfg : :obj:`dict`
        A config dictionary.

    Returns
    -------
    :obj:`tuple` of :obj:`list` of :obj:`ColorImage`, :obj:`list` of :obj:`DepthImage`, :obj:`CameraIntrinsics`
        A set of ColorImages and DepthImages, and the Kinect's CameraIntrinsics
        for its IR sensor.
    """
    if 'prestored_data' in cfg.keys() and cfg['prestored_data'] == 1:
        sensor = VirtualKinect2Sensor(path_to_images=cfg['prestored_data_dir'], frame=cfg['sensor']['frame'])
    else:
        sensor = Kinect2Sensor(device_num=cfg['sensor']['device_num'], frame=cfg['sensor']['frame'],
                               packet_pipeline_mode=cfg['sensor']['pipeline_mode'])
    sensor.start()
    ir_intrinsics = sensor.ir_intrinsics

    # get raw images
    colors = []
    depths = []

    for _ in range(cfg['num_images']):
        color, depth, _ = sensor.frames()
        colors.append(color)
        depths.append(depth)

    sensor.stop()

    return colors, depths, ir_intrinsics
