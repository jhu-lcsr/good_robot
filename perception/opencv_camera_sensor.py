'''
Abstraction for interacting with video devices that interface with OpenCV
Author: Jacky Liang
'''
import cv2
import numpy as np
from time import time

from .camera_sensor import CameraSensor
from .image import ColorImage

class OpenCVCameraSensor(CameraSensor):

    def __init__(self, device_id, upside_down=False):
        self._device_id = device_id
        self._upside_down = upside_down

    def start(self):
        """ Starts the OpenCVCameraSensor Stream
        Raises:
            Exception if unable to open stream
        """
        self._sensor = cv2.VideoCapture(self._device_id)
        if not self._sensor.isOpened():
            raise Exception("Unable to open OpenCVCameraSensor for id {0}".format(self._device_id))
        self.flush()

    def flush(self):
        for _ in range(6):
            self._sensor.read()

    def stop(self):
        """ Stops the OpenCVCameraSensor Stream """
        self._sensor.release()

    def frames(self, flush=True):
        """ Returns the latest color image from the stream
        Raises:
            Exception if opencv sensor gives ret_val of 0
        """
        self.flush()
        ret_val, frame = self._sensor.read()
        if not ret_val:
            raise Exception("Unable to retrieve frame from OpenCVCameraSensor for id {0}".format(self._device_id))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._upside_down:
            frame = np.flipud(frame).astype(np.uint8)
            frame = np.fliplr(frame).astype(np.uint8)
        return ColorImage(frame)
