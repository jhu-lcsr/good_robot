import logging
import numpy as np
import os
import time

from autolab_core import RigidTransform, PointCloud
from . import CameraSensor, ColorImage, CameraIntrinsics, PhoXiSensor, WebcamSensor

class ColorizedPhoXiSensor(CameraSensor):
    """Class for using a Logitech Webcam sensor to colorize a PhoXi's point clouds.
    """

    def __init__(self, phoxi_config, webcam_config, calib_dir, frame='phoxi'):
        """Initialize a webcam-colorized PhoXi sensor.

        Parameters
        ----------
        frame : str
            A name for the frame in which images are returned.
        phoxi_config : dict
            Config for the PhoXi camera.
        phoxi_to_world_fn : str
            Filepath for T_phoxi_world.
        webcam_config : dict
            Config for the webcam.
        webcam_to_world_fn : str
            Filepath for T_webcam_world
        """
        self._frame = frame
        phoxi_to_world_fn = os.path.join(calib_dir, 'phoxi', 'phoxi_to_world.tf')
        webcam_to_world_fn = os.path.join(calib_dir, 'webcam', 'webcam_to_world.tf')
        self._T_phoxi_world = RigidTransform.load(phoxi_to_world_fn)
        self._T_webcam_world = RigidTransform.load(webcam_to_world_fn)
        self._phoxi = PhoXiSensor(**phoxi_config)
        self._webcam = WebcamSensor(**webcam_config)
        self._camera_intr = self._phoxi.ir_intrinsics
        self._running = False

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
        running = self._webcam.start()
        if not running:
            return running

        running &= self._phoxi.start()
        if not running:
            self._webcam.stop()
        return running

    def stop(self):
        """Stop the sensor.
        """
        # Check that everything is running
        if not self._running:
            logging.warning('Colorized PhoXi not running. Aborting stop')
            return False

        self._webcam.stop()
        self._phoxi.stop()

        return True

    def frames(self):
        """Retrieve a new frame from the PhoXi and convert it to a ColorImage,
        a DepthImage, and an IrImage.

        Returns
        -------
        :obj:`tuple` of :obj:`ColorImage`, :obj:`DepthImage`, :obj:`IrImage`, :obj:`numpy.ndarray`
            The ColorImage, DepthImage, and IrImage of the current frame.
        """
        _, phoxi_depth_im, _ = self._phoxi.frames()
        webcam_color_im, _, _ = self._webcam.frames(most_recent=True)

        # Colorize PhoXi Image
        phoxi_color_im = self._colorize(phoxi_depth_im, webcam_color_im)
        return phoxi_color_im, phoxi_depth_im, None

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

    def _colorize(self, depth_im, color_im):
        """Colorize a depth image from the PhoXi using a color image from the webcam.

        Parameters
        ----------
        depth_im : DepthImage
            The PhoXi depth image.
        color_im : ColorImage
            Corresponding color image.

        Returns
        -------
        ColorImage
            A colorized image corresponding to the PhoXi depth image.
        """
        # Project the point cloud into the webcam's frame
        target_shape = (depth_im.data.shape[0], depth_im.data.shape[1], 3)
        pc_depth = self._phoxi.ir_intrinsics.deproject(depth_im)
        pc_color = self._T_webcam_world.inverse().dot(self._T_phoxi_world).apply(pc_depth)

        # Sort the points by their distance from the webcam's apeture
        pc_data = pc_color.data.T
        dists = np.linalg.norm(pc_data, axis=1)
        order = np.argsort(dists)
        pc_data = pc_data[order]
        pc_color = PointCloud(pc_data.T, frame=self._webcam.color_intrinsics.frame)
        sorted_dists = dists[order]
        sorted_depths = depth_im.data.flatten()[order]

        # Generate image coordinates for each sorted point
        icds = self._webcam.color_intrinsics.project(pc_color).data.T

        # Create mask for points that are masked by others
        rounded_icds = np.array(icds / 3.0, dtype=np.uint32)
        unique_icds, unique_inds, unique_inv = np.unique(rounded_icds, axis=0, return_index=True, return_inverse=True)
        icd_depths = sorted_dists[unique_inds]
        min_depths_pp = icd_depths[unique_inv]
        depth_delta_mask = np.abs(min_depths_pp - sorted_dists) < 5e-3

        # Create mask for points with missing depth or that lie outside the image
        valid_mask = np.logical_and(np.logical_and(icds[:,0] >= 0, icds[:,0] < self._webcam.color_intrinsics.width),
                                    np.logical_and(icds[:,1] >= 0, icds[:,1] < self._webcam.color_intrinsics.height))
        valid_mask = np.logical_and(valid_mask, sorted_depths != 0.0)
        valid_mask = np.logical_and(valid_mask, depth_delta_mask)
        valid_icds = icds[valid_mask]

        colors = color_im.data[valid_icds[:,1],valid_icds[:,0],:]
        color_im_data = np.zeros((target_shape[0] * target_shape[1], target_shape[2]), dtype=np.uint8)
        color_im_data[valid_mask] = colors
        color_im_data[order] = color_im_data.copy()
        color_im_data = color_im_data.reshape(target_shape)
        return ColorImage(color_im_data, frame=self._frame)


