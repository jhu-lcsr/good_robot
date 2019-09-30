"""
Encapsulates camera intrinsic parameters for projecting / deprojecitng points
Author: Jeff Mahler
"""
import copy
import logging
import numpy as np
import json
import os

from autolab_core import Point, PointCloud, ImageCoords

from .constants import INTR_EXTENSION
from .image import DepthImage, PointCloudImage

class CameraIntrinsics(object):
    """A set of intrinsic parameters for a camera. This class is used to project
    and deproject points.
    """

    def __init__(self, frame, fx, fy=None, cx=0.0, cy=0.0, skew=0.0, height=None, width=None):
        """Initialize a CameraIntrinsics model.

        Parameters
        ----------
        frame : :obj:`str`
            The frame of reference for the point cloud.
        fx : float
            The x-axis focal length of the camera in pixels.
        fy : float
            The y-axis focal length of the camera in pixels.
        cx : float
            The x-axis optical center of the camera in pixels.
        cy : float
            The y-axis optical center of the camera in pixels.
        skew : float
            The skew of the camera in pixels.
        height : float
            The height of the camera image in pixels.
        width : float
            The width of the camera image in pixels
        """
        self._frame = frame
        self._fx = float(fx)
        self._fy = float(fy)
        self._cx = float(cx)
        self._cy = float(cy)
        self._skew = float(skew)
        self._height = int(height)
        self._width = int(width)

        # set focal, camera center automatically if under specified
        if fy is None:
            self._fy = fx

        # set camera projection matrix
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [       0,   self._fy, self._cy],
                            [       0,          0,        1]])

    @property
    def frame(self):
        """:obj:`str` : The frame of reference for the point cloud.
        """
        return self._frame

    @property
    def fx(self):
        """float : The x-axis focal length of the camera in pixels.
        """
        return self._fx

    @property
    def fy(self):
        """float : The y-axis focal length of the camera in pixels.
        """
        return self._fy

    @property
    def cx(self):
        """float : The x-axis optical center of the camera in pixels.
        """
        return self._cx

    @cx.setter
    def cx(self, z):
        self._cx = z
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [       0,   self._fy, self._cy],
                            [       0,          0,        1]])

    @property
    def cy(self):
        """float : The y-axis optical center of the camera in pixels.
        """
        return self._cy

    @cy.setter
    def cy(self, z):
        self._cy = z
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [       0,   self._fy, self._cy],
                            [       0,          0,        1]])

    @property
    def skew(self):
        """float : The skew of the camera in pixels.
        """
        return self._skew

    @property
    def height(self):
        """float : The height of the camera image in pixels.
        """
        return self._height

    @property
    def width(self):
        """float : The width of the camera image in pixels
        """
        return self._width

    @property
    def proj_matrix(self):
        """:obj:`numpy.ndarray` : The 3x3 projection matrix for this camera.
        """
        return self._K

    @property
    def K(self):
        """:obj:`numpy.ndarray` : The 3x3 projection matrix for this camera.
        """
        return self._K

    @property
    def vec(self):
        """:obj:`numpy.ndarray` : Vector representation for this camera.
        """
        return np.r_[self.fx, self.fy, self.cx, self.cy, self.skew, self.height, self.width]
    
    @property
    def rosmsg(self):
        """:obj:`sensor_msgs.CamerInfo` : Returns ROS CamerInfo msg 
        """
        from sensor_msgs.msg import CameraInfo, RegionOfInterest
        from std_msgs.msg import Header

        msg_header = Header()
        msg_header.frame_id = self._frame

        msg_roi = RegionOfInterest()
        msg_roi.x_offset = 0
        msg_roi.y_offset = 0
        msg_roi.height = 0
        msg_roi.width = 0
        msg_roi.do_rectify = 0

        msg = CameraInfo()
        msg.header = msg_header
        msg.height = self._height
        msg.width = self._width
        msg.distortion_model = 'plumb_bob'
        msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.K = [self._fx, 0.0, self._cx, 0.0, self._fy, self._cy, 0.0, 0.0, 1.0]
        msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.P = [self._fx, 0.0, self._cx, 0.0, 0.0, self._fx, self._cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        msg.binning_x = 0
        msg.binning_y = 0
        msg.roi = msg_roi

        return msg

    @staticmethod
    def from_vec(vec, frame='unassigned'):
        return CameraIntrinsics(frame,
                                fx=vec[0],
                                fy=vec[1],
                                cx=vec[2],
                                cy=vec[3],
                                skew=vec[4],
                                height=vec[5],
                                width=vec[6])
    
    def crop(self, height, width, crop_ci, crop_cj):
        """ Convert to new camera intrinsics for crop of image from original camera.

        Parameters
        ----------
        height : int
            height of crop window
        width : int
            width of crop window
        crop_ci : int
            row of crop window center
        crop_cj : int
            col of crop window center

        Returns
        -------
        :obj:`CameraIntrinsics`
            camera intrinsics for cropped window
        """
        cx = self.cx + float(width-1)/2 - crop_cj
        cy = self.cy + float(height-1)/2 - crop_ci
        cropped_intrinsics = CameraIntrinsics(frame=self.frame,
                                              fx=self.fx,
                                              fy=self.fy,
                                              skew=self.skew,
                                              cx=cx, cy=cy,
                                              height=height,
                                              width=width)
        return cropped_intrinsics

    def resize(self, scale):
        """ Convert to new camera intrinsics with parameters for resized image.
        
        Parameters
        ----------
        scale : float
            the amount to rescale the intrinsics
        
        Returns
        -------
        :obj:`CameraIntrinsics`
            camera intrinsics for resized image        
        """
        center_x = float(self.width-1) / 2
        center_y = float(self.height-1) / 2
        orig_cx_diff = self.cx - center_x
        orig_cy_diff = self.cy - center_y
        height = scale * self.height
        width = scale * self.width
        scaled_center_x = float(width-1) / 2
        scaled_center_y = float(height-1) / 2
        fx = scale * self.fx
        fy = scale * self.fy
        skew = scale * self.skew
        cx = scaled_center_x + scale * orig_cx_diff
        cy = scaled_center_y + scale * orig_cy_diff
        scaled_intrinsics = CameraIntrinsics(frame=self.frame,
                                              fx=fx, fy=fy, skew=skew, cx=cx, cy=cy,
                                              height=height, width=width)
        return scaled_intrinsics

    def project(self, point_cloud, round_px=True):
        """Projects a point cloud onto the camera image plane.

        Parameters
        ----------
        point_cloud : :obj:`autolab_core.PointCloud` or :obj:`autolab_core.Point`
            A PointCloud or Point to project onto the camera image plane.

        round_px : bool
            If True, projections are rounded to the nearest pixel.

        Returns
        -------
        :obj:`autolab_core.ImageCoords` or :obj:`autolab_core.Point`
            A corresponding set of image coordinates representing the given
            PointCloud's projections onto the camera image plane. If the input
            was a single Point, returns a 2D Point in the camera plane.

        Raises
        ------
        ValueError
            If the input is not a PointCloud or Point in the same reference
            frame as the camera.
        """
        if not isinstance(point_cloud, PointCloud) and not (isinstance(point_cloud, Point) and point_cloud.dim == 3):
            raise ValueError('Must provide PointCloud or 3D Point object for projection')
        if point_cloud.frame != self._frame:
            raise ValueError('Cannot project points in frame %s into camera with frame %s' %(point_cloud.frame, self._frame))

        points_proj = self._K.dot(point_cloud.data)
        if len(points_proj.shape) == 1:
            points_proj = points_proj[:, np.newaxis]
        point_depths = np.tile(points_proj[2,:], [3, 1])
        points_proj = np.divide(points_proj, point_depths)
        if round_px:
            points_proj = np.round(points_proj)

        if isinstance(point_cloud, Point):
            return Point(data=points_proj[:2,:].astype(np.int16), frame=self._frame)
        return ImageCoords(data=points_proj[:2,:].astype(np.int16), frame=self._frame)

    def project_to_image(self, point_cloud, round_px=True):
        """Projects a point cloud onto the camera image plane and creates
        a depth image. Zero depth means no point projected into the camera
        at that pixel location (i.e. infinite depth).

        Parameters
        ----------
        point_cloud : :obj:`autolab_core.PointCloud` or :obj:`autolab_core.Point`
            A PointCloud or Point to project onto the camera image plane.

        round_px : bool
            If True, projections are rounded to the nearest pixel.

        Returns
        -------
        :obj:`DepthImage`
            A DepthImage generated from projecting the point cloud into the
            camera.

        Raises
        ------
        ValueError
            If the input is not a PointCloud or Point in the same reference
            frame as the camera.
        """
        if not isinstance(point_cloud, PointCloud) and not (isinstance(point_cloud, Point) and point_cloud.dim == 3):
            raise ValueError('Must provide PointCloud or 3D Point object for projection')
        if point_cloud.frame != self._frame:
            raise ValueError('Cannot project points in frame %s into camera with frame %s' %(point_cloud.frame, self._frame))

        points_proj = self._K.dot(point_cloud.data)
        if len(points_proj.shape) == 1:
            points_proj = points_proj[:, np.newaxis]
        point_depths = points_proj[2,:]
        point_z = np.tile(point_depths, [3, 1])
        points_proj = np.divide(points_proj, point_z)
        if round_px:
            points_proj = np.round(points_proj)
        points_proj = points_proj[:2,:].astype(np.int16)

        valid_ind = np.where((points_proj[0,:] >= 0) & \
                             (points_proj[1,:] >= 0) & \
                             (points_proj[0,:] < self.width) & \
                             (points_proj[1,:] < self.height))[0]

        depth_data = np.zeros([self.height, self.width])
        depth_data[points_proj[1,valid_ind], points_proj[0,valid_ind]] = point_depths[valid_ind]
        return DepthImage(depth_data, frame=self.frame)

    def deproject(self, depth_image):
        """Deprojects a DepthImage into a PointCloud.

        Parameters
        ----------
        depth_image : :obj:`DepthImage`
            The 2D depth image to projet into a point cloud.

        Returns
        -------
        :obj:`autolab_core.PointCloud`
            A 3D point cloud created from the depth image.

        Raises
        ------
        ValueError
            If depth_image is not a valid DepthImage in the same reference frame
            as the camera.
        """
        # check valid input
        if not isinstance(depth_image, DepthImage):
            raise ValueError('Must provide DepthImage object for projection')
        if depth_image.frame != self._frame:
            raise ValueError('Cannot deproject points in frame %s from camera with frame %s' %(depth_image.frame, self._frame))

        # create homogeneous pixels
        row_indices = np.arange(depth_image.height)
        col_indices = np.arange(depth_image.width)
        pixel_grid = np.meshgrid(col_indices, row_indices)
        pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
        pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
        depth_arr = np.tile(depth_image.data.flatten(), [3,1])

        # deproject
        points_3d = depth_arr * np.linalg.inv(self._K).dot(pixels_homog)
        return PointCloud(data=points_3d, frame=self._frame)

    def deproject_to_image(self, depth_image):
        """Deprojects a DepthImage into a PointCloudImage.

        Parameters
        ----------
        depth_image : :obj:`DepthImage`
            The 2D depth image to projet into a point cloud.

        Returns
        -------
        :obj:`PointCloudImage`
            A point cloud image created from the depth image.

        Raises
        ------
        ValueError
            If depth_image is not a valid DepthImage in the same reference frame
            as the camera.
        """
        point_cloud = self.deproject(depth_image)
        point_cloud_im_data = point_cloud.data.T.reshape(depth_image.height, depth_image.width, 3)
        return PointCloudImage(data=point_cloud_im_data,
                               frame=self._frame)

    def deproject_pixel(self, depth, pixel):
        """Deprojects a single pixel with a given depth into a 3D point.

        Parameters
        ----------
        depth : float
            The depth value at the given pixel location.

        pixel : :obj:`autolab_core.Point`
            A 2D point representing the pixel's location in the camera image.

        Returns
        -------
        :obj:`autolab_core.Point`
            The projected 3D point.

        Raises
        ------
        ValueError
            If pixel is not a valid 2D Point in the same reference frame
            as the camera.
        """
        if not isinstance(pixel, Point) and not pixel.dim == 2:
            raise ValueError('Must provide 2D Point object for pixel projection')
        if pixel.frame != self._frame:
            raise ValueError('Cannot deproject pixel in frame %s from camera with frame %s' %(pixel.frame, self._frame))

        point_3d = depth * np.linalg.inv(self._K).dot(np.r_[pixel.data, 1.0])
        return Point(data=point_3d, frame=self._frame)

    def save(self, filename):
        """Save the CameraIntrinsics object to a .intr file.

        Parameters
        ----------
        filename : :obj:`str`
            The .intr file to save the object to.

        Raises
        ------
        ValueError
            If filename does not have the .intr extension.
        """
        file_root, file_ext = os.path.splitext(filename)
        if file_ext.lower() != INTR_EXTENSION:
            raise ValueError('Extension %s not supported for CameraIntrinsics. Must be stored with extension %s' %(file_ext, INTR_EXTENSION))

        camera_intr_dict = copy.deepcopy(self.__dict__)
        camera_intr_dict['_K'] = 0 # can't save matrix
        f = open(filename, 'w')
        json.dump(camera_intr_dict, f)
        f.close()

    @staticmethod
    def load(filename):
        """Load a CameraIntrinsics object from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The .intr file to load the object from.

        Returns
        -------
        :obj:`CameraIntrinsics`
            The CameraIntrinsics object loaded from the file.

        Raises
        ------
        ValueError
            If filename does not have the .intr extension.
        """
        file_root, file_ext = os.path.splitext(filename)
        if file_ext.lower() != INTR_EXTENSION:
            raise ValueError('Extension %s not supported for CameraIntrinsics. Must be stored with extension %s' %(file_ext, INTR_EXTENSION))

        f = open(filename, 'r')
        ci = json.load(f)
        f.close()
        return CameraIntrinsics(frame=ci['_frame'],
                                fx=ci['_fx'],
                                fy=ci['_fy'],
                                cx=ci['_cx'],
                                cy=ci['_cy'],
                                skew=ci['_skew'],
                                height=ci['_height'],
                                width=ci['_width'])
