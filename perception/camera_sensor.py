"""
Abstract class for Camera sensors.
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

from .camera_intrinsics import CameraIntrinsics
from .constants import *
from .image import ColorImage, DepthImage, IrImage

import os

class CameraSensor(object):
    """Abstract base class for camera sensors.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def start(self):
        """Starts the sensor stream.
        """
        pass

    @abstractmethod
    def stop(self):
        """Stops the sensor stream.
        """
        pass

    def reset(self):
        """Restarts the sensor stream.
        """
        self.stop()
        self.start()

    @abstractmethod
    def frames(self):
        """Returns the latest set of frames.
        """
        pass


class VirtualSensor(CameraSensor):
    SUPPORTED_FILE_EXTS = ['.png', '.npy']

    """ Class for a virtual sensor that uses pre-captured images
    stored to disk instead of actually connecting to a sensor.
    For debugging purposes.
    """
    def __init__(self, path_to_images, frame=None, loop=True):
        """Create a new virtualized Primesense sensor.

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
        loop : :obj:`str`
            Whether or not to loop back to the first image after running out
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
        self._color_ext = '.png'
        for filename in filenames:
            file_root, file_ext = os.path.splitext(filename)
            color_ind = file_root.rfind('color')

            if file_ext in VirtualSensor.SUPPORTED_FILE_EXTS \
               and color_ind != -1:
                self._color_ext = file_ext

            if self._frame is None and file_ext == INTR_EXTENSION and color_ind != -1:
                self._frame = file_root[:color_ind-1]
                self._color_frame = file_root
                self._ir_frame = file_root

        # load color intrinsics
        color_intr_filename = os.path.join(self._path_to_images, '%s_color.intr' %(self._frame))
        ir_intr_filename = os.path.join(self._path_to_images, '%s_ir.intr' %(self._frame))
        generic_intr_filename = os.path.join(self._path_to_images, '%s.intr' %(self._frame))
        if os.path.exists(color_intr_filename):
            self._color_intr = CameraIntrinsics.load(color_intr_filename)
        else:
            self._color_intr = CameraIntrinsics.load(generic_intr_filename)
        if os.path.exists(ir_intr_filename):            
            self._ir_intr = CameraIntrinsics.load(ir_intr_filename)
        else:
            self._ir_intr = CameraIntrinsics.load(generic_intr_filename)            

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
        """:obj:`CameraIntrinsics` : The camera intrinsics for the sensor's color camera.
        """
        return self._color_intr

    @property
    def ir_intrinsics(self):
        """:obj:`CameraIntrinsics` : The camera intrinsics for the sensor's IR camera.
        """
        return self._ir_intr


    def start(self):
        """Starts the sensor stream.

        In this virtualized sensor, this simply resets the image index to zero.
        Everytime start is called, we start the stream again at the first image.
        """
        self._im_index = 0
        self._running = True

    def stop(self):
        """Stops the sensor stream.

        Returns
        -------
        bool
            True if the stream was stopped, False if the device was already
            stopped or was not otherwise available.
        """
        if not self._running:
            return False
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
            If the stream is not running or if all images in the
            directory have been used.
        """
        if not self._running:
            raise RuntimeError('Device pointing to %s not runnning. Cannot read frames' %(self._path_to_images))

        if self._im_index >= self._num_images:
            raise RuntimeError('Device is out of images')

        # read images
        color_filename = os.path.join(self._path_to_images, 'color_%d%s' %(self._im_index, self._color_ext))
        color_im = ColorImage.open(color_filename, frame=self._frame)
        depth_filename = os.path.join(self._path_to_images, 'depth_%d.npy' %(self._im_index))
        depth_im = DepthImage.open(depth_filename, frame=self._frame)
        self._im_index = (self._im_index + 1) % self._num_images
        return color_im, depth_im, None

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
    

class TensorDatasetVirtualSensor(VirtualSensor):
    CAMERA_INTR_FIELD = 'camera_intrs'
    COLOR_IM_FIELD = 'color_ims'
    DEPTH_IM_FIELD = 'depth_ims'
    IMAGE_FIELDS = [COLOR_IM_FIELD, DEPTH_IM_FIELD]
    
    """ Class for a virtual sensor that runs off of images stored in a
    tensor dataset.
    """
    def __init__(self, dataset_path, frame=None, loop=True):
        self._dataset_path = dataset_path
        self._frame = frame
        self._color_frame = frame
        self._ir_frame = frame
        self._im_index = 0
        self._running = False
        
        from dexnet.learning import TensorDataset
        self._dataset = TensorDataset.open(self._dataset_path)
        self._num_images = self._dataset.num_datapoints
        self._image_rescale_factor = 1.0
        if 'image_rescale_factor' in self._dataset.metadata.keys():
            self._image_rescale_factor = 1.0 / self._dataset.metadata['image_rescale_factor']
        
        datapoint = self._dataset.datapoint(0, [TensorDatasetVirtualSensor.CAMERA_INTR_FIELD])
        camera_intr_vec = datapoint[TensorDatasetVirtualSensor.CAMERA_INTR_FIELD]
        self._color_intr = CameraIntrinsics.from_vec(camera_intr_vec, frame=self._color_frame).resize(self._image_rescale_factor)
        self._ir_intr = CameraIntrinsics.from_vec(camera_intr_vec, frame=self._ir_frame).resize(self._image_rescale_factor)

    def frames(self):
        """Retrieve the next frame from the tensor dataset and convert it to a ColorImage,
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
            If the stream is not running or if all images in the
            directory have been used.
        """
        if not self._running:
            raise RuntimeError('Device pointing to %s not runnning. Cannot read frames' %(self._path_to_images))

        if self._im_index >= self._num_images:
            raise RuntimeError('Device is out of images')

        # read images
        datapoint = self._dataset.datapoint(self._im_index,
                                            TensorDatasetVirtualSensor.IMAGE_FIELDS)
        color_im = ColorImage(datapoint[TensorDatasetVirtualSensor.COLOR_IM_FIELD],
                              frame=self._frame)
        depth_im = DepthImage(datapoint[TensorDatasetVirtualSensor.DEPTH_IM_FIELD],
                              frame=self._frame)
        if self._image_rescale_factor != 1.0:
            color_im = color_im.resize(self._image_rescale_factor)
            depth_im = depth_im.resize(self._image_rescale_factor, interp='nearest')
        self._im_index = (self._im_index + 1) % self._num_images
        return color_im, depth_im, None
