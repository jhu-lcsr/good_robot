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

class OrthographicIntrinsics(object):
    """A set of intrinsic parameters for orthographic point cloud projections
    """

    def __init__(self, frame,
                 vol_width,
                 vol_height,
                 vol_depth,
                 plane_height,
                 plane_width,
                 depth_scale=1.0):
        """Initialize a CameraIntrinsics model.

        Parameters
        ----------
        frame : :obj:`str`
            The frame of reference for the point cloud.
        vol_height : float
            The height of the 3D projection volume in meters.
        vol_width : float
            The width of the 3D projection volume in meters.
        vol_depth : float
            The depth of the 3D projection volume in meters.
        plane_height : float
            The height of the projection plane in pixels.
        plane_width : float
            The width of the projection plane in pixels.
        depth_scale : float
            The scale of the depth values.
        """
        self._frame = frame
        self._vol_height = float(vol_height)
        self._vol_width = float(vol_width)
        self._vol_depth = float(vol_depth)
        self._plane_height = float(plane_height)
        self._plane_width = float(plane_width)
        self._depth_scale = float(depth_scale)

    @property
    def frame(self):
        """:obj:`str` : The frame of reference for the point cloud.
        """
        return self._frame

    @property
    def plane_height(self):
        """float : The height of the projection plane in pixels.
        """
        return self._height

    @property
    def plane_width(self):
        """float : The width of the projection plane in pixels.
        """
        return self._plane_width

    @property
    def S(self):
        """:obj:`numpy.ndarray` : The 3x3 scaling matrix for this projection
        """
        S = np.array([[self._plane_width / self._vol_width, 0, 0],
                      [0, self._plane_height / self._vol_height, 0],
                      [0, 0, self._depth_scale / self._vol_depth]])
        return S

    @property
    def t(self):
        """:obj:`numpy.ndarray` : The 3x1 translation matrix for this projection
        """
        t = np.array([self._plane_width / 2,
                      self._plane_height / 2,
                      self._depth_scale / 2])
        return t
    
    @property
    def proj_matrix(self):
        """:obj:`numpy.ndarray` : The 4x4 projection matrix for this camera.
        """
        return self.P

    @property
    def P(self):
        """:obj:`numpy.ndarray` : The 4x4 projection matrix for this camera.
        """
        P = np.r_[np.c_[self.S, self.t], np.array([0,0,0,1])]
        return P

    def project(self, point_cloud, round_px=True):
        """Projects a point cloud onto the projection plane.

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

        points_proj = self.S.dot(point_cloud.data) + self.t
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

        points_proj = self.S.dot(point_cloud.data) + self.t
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
        depth_data = depth_image.data.flatten()
        pixels_homog = np.r_[pixels, depth_data.reshape(1, depth_data.shape[0])]

        # deproject
        points_3d = np.linalg.inv(self.S).dot(pixels_homog - np.tile(self.t.reshape(3,1), [1, pixels_homog.shape[1]]))
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

        point = np.r_[pixel.data, depth]
        point_3d = np.linalg.inv(self.S).dot(point - self.t)
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
            raise ValueError('Extension %s not supported for OrhtographicIntrinsics. Must be stored with extension %s' %(file_ext, INTR_EXTENSION))

        camera_intr_dict = copy.deepcopy(self.__dict__)
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
        return OrthographicIntrinsics(frame=ci['_frame'],
                                      vol_height=ci['_vol_height'],
                                      vol_width=ci['_vol_width'],
                                      vol_depth=ci['_vol_depth'],
                                      plane_height=ci['_plane_height'],
                                      plane_width=ci['_plane_width'],
                                      depth_scale=ci['_depth_scale'])
