import cv2
import logging
import numpy as np
import os
import subprocess
import shlex

from . import CameraSensor, ColorImage, CameraIntrinsics

class WebcamSensor(CameraSensor):
    """Class for interfacing with a Logitech webcam sensor (720p).
    """

    def __init__(self, frame='webcam', device_id=0):
        """Initialize a Logitech webcam sensor.

        Parameters
        ----------
        frame : str
            A name for the frame in which RGB images are returned.
        device_id : int
            The device ID for the webcam (by default, zero).
        """
        self._frame = frame
        self._camera_intr = None
        self._device_id = device_id
        self._cap = None
        self._running = False
        self._adjust_exposure = True

        # Set up camera intrinsics for the sensor
        width, height = 1280, 960
        focal_x, focal_y = 1430., 1420.
        center_x, center_y = 1280/2, 960/2
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

    def start(self):
        """Start the sensor.
        """
        self._cap = cv2.VideoCapture(self._device_id + cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._running = False
            self._cap.release()
            self._cap = None
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_intr.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_intr.height)
        self._running = True

        # Capture 5 frames to flush webcam sensor
        for _ in range(5):
            _ = self.frames()

        return True

    def stop(self):
        """Stop the sensor.
        """
        # Check that everything is running
        if not self._running:
            logging.warning('Webcam not running. Aborting stop')
            return False

        if self._cap:
            self._cap.release()
            self._cap = None
        self._running = False

        return True

    def frames(self, most_recent=False):
        """Retrieve a new frame from the PhoXi and convert it to a ColorImage,
        a DepthImage, and an IrImage.

        Parameters
        ----------
        most_recent: bool
            If true, the OpenCV buffer is emptied for the webcam before reading the most recent frame.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`, :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the current frame.
        """
        if most_recent:
            for i in xrange(4):
                self._cap.grab()
        for i in range(1):
            if self._adjust_exposure:
                try:
                    command = 'v4l2-ctl -d /dev/video{} -c exposure_auto=1 -c exposure_auto_priority=0 -c exposure_absolute=100 -c saturation=60 -c gain=140'.format(self._device_id)
                    FNULL = open(os.devnull, 'w')
                    subprocess.call(shlex.split(command), stdout=FNULL, stderr=subprocess.STDOUT)
                except:
                    pass
            ret, frame = self._cap.read()
        rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return ColorImage(rgb_data, frame=self._frame), None, None
