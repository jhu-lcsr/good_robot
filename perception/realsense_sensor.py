"""
Class for interfacing with the Intel RealSense D400-Series.
"""
import logging
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    logging.warning('Unable to import pyrealsense2.')

from perception import CameraIntrinsics, CameraSensor, ColorImage, DepthImage

class RealSenseRegistrationMode:
    """Realsense registration mode.
    """
    NONE = 0
    DEPTH_TO_COLOR = 1


class RealSenseSensor(CameraSensor):
    """Class for interacting with a RealSense D400-series sensor.

    pyrealsense2 should be installed from source with the following
    commands:

    >>> git clone https://github.com/IntelRealSense/librealsense
    >>> cd librealsense
    >>> mkdir build
    >>> cd build
    >>> cmake .. \
        -DBUILD_EXAMPLES=true \
        -DBUILD_WITH_OPENMP=false \
        -DHWM_OVER_XU=false \
        -DBUILD_PYTHON_BINDINGS=true \
        -DPYTHON_EXECUTABLE:FILEPATH=/path/to/your/python/library/ \
        -G Unix\ Makefiles
    >>> make -j4
    >>> sudo make install
    >>> export PYTHONPATH=$PYTHONPATH:/usr/local/lib
    """
    COLOR_IM_HEIGHT = 480
    COLOR_IM_WIDTH = 640
    DEPTH_IM_HEIGHT = 480
    DEPTH_IM_WIDTH = 640
    FPS = 30

    def __init__(self,
                 cam_id,
                 filter_depth=True,
                 frame=None,
                 registration_mode=RealSenseRegistrationMode.DEPTH_TO_COLOR):
        self._running = None

        self.id = cam_id
        self._registration_mode = registration_mode
        self._filter_depth = filter_depth

        self._frame = frame

        if self._frame is None:
            self._frame = 'realsense'
        self._color_frame = '%s_color' % (self._frame)

        # realsense objects
        self._pipe = rs.pipeline()
        self._cfg = rs.config()
        self._align = rs.align(rs.stream.color)

        # camera parameters
        self._depth_scale = None
        self._intrinsics = np.eye(3)

        # post-processing filters
        self._colorizer = rs.colorizer()
        self._spatial_filter = rs.spatial_filter()
        self._hole_filling = rs.hole_filling_filter()

    def _config_pipe(self):
        """Configures the pipeline to stream color and depth.
        """
        self._cfg.enable_device(self.id)

        # configure the color stream
        self._cfg.enable_stream(
            rs.stream.color,
            RealSenseSensor.COLOR_IM_WIDTH,
            RealSenseSensor.COLOR_IM_HEIGHT,
            rs.format.bgr8,
            RealSenseSensor.FPS
        )

        # configure the depth stream
        self._cfg.enable_stream(
            rs.stream.depth,
            RealSenseSensor.DEPTH_IM_WIDTH,
            360 if self._depth_align else RealSenseSensor.DEPTH_IM_HEIGHT,
            rs.format.z16,
            RealSenseSensor.FPS
        )

    def _set_depth_scale(self):
        """Retrieve the scale of the depth sensor.
        """
        sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = sensor.get_depth_scale()

    def _set_intrinsics(self):
        """Read the intrinsics matrix from the stream.
        """
        strm = self._profile.get_stream(rs.stream.color)
        obj = strm.as_video_stream_profile().get_intrinsics()
        self._intrinsics[0, 0] = obj.fx
        self._intrinsics[1, 1] = obj.fy
        self._intrinsics[0, 2] = obj.ppx
        self._intrinsics[1, 2] = obj.ppy

    @property
    def color_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the RealSense color camera.
        """
        return CameraIntrinsics(
            self._frame,
            self._intrinsics[0, 0],
            self._intrinsics[1, 1],
            self._intrinsics[0, 2],
            self._intrinsics[1, 2],
            height=RealSenseSensor.COLOR_IM_HEIGHT,
            width=RealSenseSensor.COLOR_IM_WIDTH,
        )

    def __del__(self):
        """Automatically stop the sensor for safety.
        """
        if self.is_running:
            self.stop()

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

    def start(self):
        """Start the sensor.
        """
        try:
            self._depth_align = False
            if self._registration_mode == RealSenseRegistrationMode.DEPTH_TO_COLOR:
                self._depth_align = True

            self._config_pipe()
            self._profile = self._pipe.start(self._cfg)

            # store intrinsics and depth scale
            self._set_depth_scale()
            self._set_intrinsics()

            # skip few frames to give auto-exposure a chance to settle
            for _ in range(5):
                self._pipe.wait_for_frames()

            self._running = True
        except RuntimeError as e:
            print(e)

    def stop(self):
        """Stop the sensor.
        """
        # check that everything is running
        if not self._running:
            logging.warning('Realsense not running. Aborting stop.')
            return False

        self._pipe.stop()
        self._running = False
        return True

    def _to_numpy(self, frame, dtype):
        arr = np.asanyarray(frame.get_data(), dtype=dtype)
        return arr

    def _filter_depth_frame(self, depth):
        out = self._spatial_filter.process(depth)
        out = self._hole_filling.process(out)
        return out

    def _read_color_and_depth_image(self):
        """Read a color and depth image from the device.
        """
        frames = self._pipe.wait_for_frames()
        if self._depth_align:
            frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            logging.warning('Could not retrieve frames.')
            return None, None

        if self._filter_depth:
            depth_frame = self._filter_depth_frame(depth_frame)

        # convert to numpy arrays
        depth_image = self._to_numpy(depth_frame, np.float32)
        color_image = self._to_numpy(color_frame, np.uint8)

        # convert depth to meters
        depth_image *= self._depth_scale

        # bgr to rgb
        color_image = color_image[..., ::-1]

        depth = DepthImage(depth_image, frame=self._frame)
        color = ColorImage(color_image, frame=self._frame)
        return color, depth

    def frames(self):
        """Retrieve a new frame from the RealSense and convert it to a ColorImage,
        a DepthImage, and an IrImage.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`, :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the current frame.

        Raises
        ------
        RuntimeError
            If the RealSense stream is not running.
        """
        color_im, depth_im = self._read_color_and_depth_image()
        return color_im, depth_im, None
