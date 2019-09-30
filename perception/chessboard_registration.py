"""
Classes for easy chessboard registration
Authors: Jeff Mahler and Jacky Liang
"""
import cv2
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import time

from autolab_core import PointCloud, RigidTransform, Point
from .image import DepthImage

class ChessboardRegistrationResult(object):
    """ Struct to encapsulate results of camera-to-chessboard registration.
    
    Attributes
    ----------
    T_camera_cb : :obj:`autolab_core.RigidTransform`
        transformation from camera to chessboard frame
    cb_points_camera : :obj:`autolab_core.PointCloud`
        3D locations of chessboard corners in the camera frame
    """
    def __init__(self, T_camera_cb, cb_points_camera):
        self.T_camera_cb = T_camera_cb
        self.cb_points_cam = cb_points_camera

class CameraChessboardRegistration:
    """
    Namespace for camera to chessboard registration functions.
    """

    @staticmethod
    def register(sensor, config):
        """
        Registers a camera to a chessboard.

        Parameters
        ----------
        sensor : :obj:`perception.RgbdSensor`
            the sensor to register
        config : :obj:`autolab_core.YamlConfig` or :obj:`dict`
            configuration file for registration

        Returns
        -------
        :obj:`ChessboardRegistrationResult`
            the result of registration

        Notes
        -----
        The config must have the parameters specified in the Other Parameters section.

        Other Parameters
        ----------------
        num_transform_avg : int
            the number of independent registrations to average together
        num_images : int
            the number of images to read for each independent registration
        corners_x : int
            the number of chessboard corners in the x-direction
        corners_y : int
            the number of chessboard corners in the y-direction
        color_image_rescale_factor : float
            amount to rescale the color image for detection (numbers around 4-8 are useful)
        vis : bool
            whether or not to visualize the registration
        """
        # read config
        num_transform_avg = config['num_transform_avg']
        num_images = config['num_images']
        sx = config['corners_x']
        sy = config['corners_y']
        point_order = config['point_order']
        color_image_rescale_factor = config['color_image_rescale_factor']
        flip_normal = config['flip_normal']
        y_points_left = False
        if 'y_points_left' in config.keys() and sx == sy:
            y_points_left = config['y_points_left']
            num_images = 1
        vis = config['vis']

        # read params from sensor
        logging.info('Registering camera %s' %(sensor.frame))
        ir_intrinsics = sensor.ir_intrinsics

        # repeat registration multiple times and average results
        R = np.zeros([3,3])
        t = np.zeros([3,1])
        points_3d_plane = PointCloud(np.zeros([3, sx*sy]), frame=sensor.ir_frame)

        k = 0
        while k < num_transform_avg:
            # average a bunch of depth images together
            depth_ims = None
            for i in range(num_images):
                start = time.time()
                small_color_im, new_depth_im, _ = sensor.frames()
                end = time.time()
                logging.info('Frames Runtime: %.3f' %(end-start))
                if depth_ims is None:
                    depth_ims = np.zeros([new_depth_im.height,
                                          new_depth_im.width,
                                          num_images])
                depth_ims[:,:,i] = new_depth_im.data

            med_depth_im = np.median(depth_ims, axis=2)
            depth_im = DepthImage(med_depth_im, sensor.ir_frame)

            # find the corner pixels in an upsampled version of the color image
            big_color_im = small_color_im.resize(color_image_rescale_factor)
            corner_px = big_color_im.find_chessboard(sx=sx, sy=sy)

            if vis:
                plt.figure()
                plt.imshow(big_color_im.data)
                for i in range(sx):
                    plt.scatter(corner_px[i,0], corner_px[i,1], s=25, c='b')
                plt.show()

            if corner_px is None:
                logging.error('No chessboard detected! Check camera exposure settings')
                continue

            # convert back to original image
            small_corner_px = corner_px / color_image_rescale_factor
        
            if vis:
                plt.figure()
                plt.imshow(small_color_im.data)
                for i in range(sx):
                    plt.scatter(small_corner_px[i,0], small_corner_px[i,1], s=25, c='b')
                plt.axis('off')
                plt.show()

            # project points into 3D
            camera_intr = sensor.ir_intrinsics
            points_3d = camera_intr.deproject(depth_im)

            # get round chessboard ind
            corner_px_round = np.round(small_corner_px).astype(np.uint16)
            corner_ind = depth_im.ij_to_linear(corner_px_round[:,0], corner_px_round[:,1])
            if corner_ind.shape[0] != sx*sy:
                logging.warning('Did not find all corners. Discarding...')
                continue

            # average 3d points
            points_3d_plane = (k * points_3d_plane + points_3d[corner_ind]) / (k + 1)
            logging.info('Registration iteration %d of %d' %(k+1, config['num_transform_avg']))
            k += 1

        # fit a plane to the chessboard corners
        X = np.c_[points_3d_plane.x_coords, points_3d_plane.y_coords, np.ones(points_3d_plane.num_points)]
        y = points_3d_plane.z_coords
        A = X.T.dot(X)
        b = X.T.dot(y)
        w = np.linalg.inv(A).dot(b)
        n = np.array([w[0], w[1], -1])
        n = n / np.linalg.norm(n)
        if flip_normal:
            n = -n
        mean_point_plane = points_3d_plane.mean()
        
        # find x-axis of the chessboard coordinates on the fitted plane
        T_camera_table = RigidTransform(translation = -points_3d_plane.mean().data,
                                    from_frame=points_3d_plane.frame,
                                    to_frame='table')
        points_3d_centered = T_camera_table * points_3d_plane

        # get points along y
        if point_order == 'row_major':
            coord_pos_x = int(math.floor(sx*sy/2.0))
            coord_neg_x = int(math.ceil(sx*sy/2.0))
            
            points_pos_x = points_3d_centered[coord_pos_x:]
            points_neg_x = points_3d_centered[:coord_neg_x]
            x_axis = np.mean(points_pos_x.data, axis=1) - np.mean(points_neg_x.data, axis=1)
            x_axis = x_axis - np.vdot(x_axis, n)*n
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(n, x_axis)
        else:
            coord_pos_y = int(math.floor(sx*(sy-1)/2.0))
            coord_neg_y = int(math.ceil(sx*(sy+1)/2.0))
            points_pos_y = points_3d_centered[:coord_pos_y]
            points_neg_y = points_3d_centered[coord_neg_y:]
            y_axis = np.mean(points_pos_y.data, axis=1) - np.mean(points_neg_y.data, axis=1)
            y_axis = y_axis - np.vdot(y_axis, n)*n
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(-n, y_axis)

        # produce translation and rotation from plane center and chessboard basis
        rotation_cb_camera = RigidTransform.rotation_from_axes(x_axis, y_axis, n)
        translation_cb_camera = mean_point_plane.data
        T_cb_camera = RigidTransform(rotation=rotation_cb_camera,
                                     translation=translation_cb_camera,
                                     from_frame='cb',
                                     to_frame=sensor.frame)
        
        if y_points_left and np.abs(T_cb_camera.y_axis[1]) > 0.1:
            if T_cb_camera.x_axis[0] > 0:
                T_cb_camera.rotation = T_cb_camera.rotation.dot(RigidTransform.z_axis_rotation(-np.pi/2).T)
            else:
                T_cb_camera.rotation = T_cb_camera.rotation.dot(RigidTransform.z_axis_rotation(np.pi/2).T)
        T_camera_cb = T_cb_camera.inverse()
                
        # optionally display cb corners with detected pose in 3d space
        if config['debug']:
            # display image with axes overlayed
            cb_center_im = camera_intr.project(Point(T_cb_camera.translation, frame=sensor.ir_frame))
            cb_x_im = camera_intr.project(Point(T_cb_camera.translation + T_cb_camera.x_axis * config['scale_amt'], frame=sensor.ir_frame))
            cb_y_im = camera_intr.project(Point(T_cb_camera.translation + T_cb_camera.y_axis * config['scale_amt'], frame=sensor.ir_frame))
            cb_z_im = camera_intr.project(Point(T_cb_camera.translation + T_cb_camera.z_axis * config['scale_amt'], frame=sensor.ir_frame))
            x_line = np.array([cb_center_im.data, cb_x_im.data])
            y_line = np.array([cb_center_im.data, cb_y_im.data])
            z_line = np.array([cb_center_im.data, cb_z_im.data])

            plt.figure()
            plt.imshow(small_color_im.data)
            plt.scatter(cb_center_im.data[0], cb_center_im.data[1])
            plt.plot(x_line[:,0], x_line[:,1], c='r', linewidth=3)
            plt.plot(y_line[:,0], y_line[:,1], c='g', linewidth=3)
            plt.plot(z_line[:,0], z_line[:,1], c='b', linewidth=3)
            plt.axis('off')
            plt.title('Chessboard frame in camera %s' %(sensor.frame))
            plt.show()

        return ChessboardRegistrationResult(T_camera_cb, points_3d_plane)
