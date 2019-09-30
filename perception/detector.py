"""
Classes for image detection
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod
import colorsys
import cv2
import IPython
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.morphology as snm

from autolab_core import Box

from .image import BinaryImage, ColorImage, DepthImage
from .object_render import RenderMode

class RgbdDetection(object):
    """ Struct to wrap the results of rgbd detection.

    Attributes
    ----------
    height : int
        height of detected object in pixels
    width : int
        width of detected object in pixels
    camera_intr : :obj:`CameraIntrinsics`
        camera intrinsics that project the detected window into the camera center
    query_im : :obj:`ColorImage`
        binary segmask for detected object as a 3-channel color image, for backwards comp
    color_im : :obj:`ColorImage`
        color thumbnail of detected object, with object masked if binary mask available
    depth_im : :obj:`DepthImage`
        depth thumbnail of detected object, with object masked if binary mask available
    binary_im : :obj:`BinaryImage`
        binary segmask of detected object
    depth_im_table : :obj:`DepthImage`
        unmasked depth image, for backwards comp
    cropped_ir_intrinsics : :obj:`CameraIntrinsics`
        alias for camera_intr, for backwards comp
    point_normal_cloud : :obj:`PointNormalCloud`
        point cloud with normals for the detected object
    """
    def __init__(self, color_thumbnail, depth_thumbnail, bounding_box, binary_thumbnail=None, camera_intr=None, contour=None):
        self.color_thumbnail = color_thumbnail         # cropped color image from bounding box
        self.depth_thumbnail = depth_thumbnail         # cropped depth image from bounding box
        self.bounding_box = bounding_box # bounding box in original source image
        self.binary_thumbnail = binary_thumbnail       # optional binary image masking the object
        self.camera_intr = camera_intr   # optional intrinsics of camera taking the thumbnail
        self.contour = contour           # optional contour describing the object boundary

    @property
    def height(self):
        return self.bounding_box.dims[0]

    @property
    def width(self):
        return self.bounding_box.dims[1]

    @property
    def query_im(self):
        return self.binary_im.to_color()

    @property
    def color_im(self):
        if self.binary_thumbnail is None:
            return self.color_thumbnail
        return self.color_thumbnail.mask_binary(self.binary_thumbnail)

    @property
    def depth_im(self):
        if self.binary_thumbnail is None:
            return self.depth_thumbnail
        return self.depth_thumbnail.mask_binary(self.binary_thumbnail)

    @property
    def binary_im(self):
        return self.binary_thumbnail

    @property
    def depth_im_table(self):
        return self.depth_thumbnail

    @property
    def cropped_ir_intrinsics(self):
        return self.camera_intr

    @property
    def virtual_camera_intrinsics(self):
        return self.camera_intr

    @property
    def point_normal_cloud(self):
        if self.camera_intr is None:
            return None
        point_normal_cloud = self.depth_thumbnail.point_normal_cloud(self.camera_intr)
        point_normal_cloud.remove_zero_points()
        return point_normal_cloud

    def image(self, render_mode):
        """ Get the image associated with a particular render mode """
        if render_mode == RenderMode.SEGMASK:
            return self.query_im
        elif render_mode == RenderMode.COLOR:
            return self.color_im
        elif render_mode == RenderMode.DEPTH:
            return self.depth_im
        else:
            raise ValueError('Render mode %s not supported' %(render_mode))

class RgbdDetector(object):
    """ Wraps methods for as many distinct objects in the image as possible.
    """
    __metaclass__ = ABCMeta    

    @abstractmethod
    def detect(self, color_im, depth_im, cfg, camera_intr=None,
               T_camera_world=None, segmask=None):
        """
        Detects all relevant objects in an rgbd image pair.

        Parameters
        ----------
        color_im : :obj:`ColorImage`
            color image for detection
        depth_im : :obj:`DepthImage`
            depth image for detection (corresponds to color image)
        cfg : :obj:`YamlConfig`
            parameters of detection function
        camera_intr : :obj:`CameraIntrinsics`
            intrinsics of the camera
        T_camera_world : :obj:`autolab_core.RigidTransform`
            registration of the camera to world frame
        segmask : :obj:`BinaryImage`
            optional segmask of invalid pixels

        Returns
        ------
        :obj:`list` of :obj:`RgbdDetection`
            all detections in the image
        """
        pass

class RgbdForegroundMaskDetector(RgbdDetector):
    """ Detect by identifying all connected components in the foreground of
    the images using background subtraction.
    """
    def detect(self, color_im, depth_im, cfg, camera_intr=None,
               T_camera_world=None, segmask=None):
        """
        Detects all relevant objects in an rgbd image pair using foreground masking.

        Parameters
        ----------
        color_im : :obj:`ColorImage`
            color image for detection
        depth_im : :obj:`DepthImage`
            depth image for detection (corresponds to color image)
        cfg : :obj:`YamlConfig`
            parameters of detection function
        camera_intr : :obj:`CameraIntrinsics`
            intrinsics of the camera
        T_camera_world : :obj:`autolab_core.RigidTransform`
            registration of the camera to world frame
        segmask : :obj:`BinaryImage`
            optional segmask of invalid pixels

        Returns
        ------
        :obj:`list` of :obj:`RgbdDetection`
            all detections in the image
        """
        # read params
        foreground_mask_tolerance = cfg['foreground_mask_tolerance']
        min_contour_area = cfg['min_contour_area']
        max_contour_area = cfg['max_contour_area']
        w = cfg['filter_dim']

        # mask image using background detection
        bgmodel = color_im.background_model()
        binary_im = color_im.foreground_mask(foreground_mask_tolerance, bgmodel=bgmodel)

        # filter the image
        y, x = np.ogrid[-w/2+1:w/2+1, -w/2+1:w/2+1]
        mask = x*x + y*y <= w/2*w/2
        filter_struct = np.zeros([w,w]).astype(np.uint8)
        filter_struct[mask] = 1
        binary_im_filtered = binary_im.apply(snm.grey_closing, structure=filter_struct)

        visualize = False
        if visualize:
            plt.figure()
            plt.imshow(binary_im_filtered.data, cmap=plt.cm.gray)
            plt.axis('off')
            plt.show()

        # find all contours
        contours = binary_im_filtered.find_contours(min_area=min_contour_area, max_area=max_contour_area)

        # convert contours to detections
        detections = []
        for contour in contours:
            box = contour.bounding_box
            color_thumbnail = color_im.crop(box.height, box.width, box.ci, box.cj)
            depth_thumbnail = depth_im.crop(box.height, box.width, box.ci, box.cj)
            binary_thumbnail = binary_im_filtered.crop(box.height, box.width, box.ci, box.cj)
            thumbnail_intr = camera_intr
            if camera_intr is not None:
                thumbnail_intr = camera_intr.crop(box.height, box.width, box.ci, box.cj)
            detections.append(RgbdDetection(color_thumbnail,
                                            depth_thumbnail,
                                            box,
                                            binary_thumbnail=binary_thumbnail,
                                            contour=contour,
                                            camera_intr=thumbnail_intr))
        return detections

class RgbdForegroundMaskQueryImageDetector(RgbdDetector):
    """ Detect by identifying all connected components in the foreground of
    the images using background subtraction.
    Converts all detections within a specified area into query images for a cnn.
    Optionally resegements the images using KMeans to remove spurious background pixels.
    """
    def _segment_color(self, color_im, bounding_box, bgmodel, cfg, vis_segmentation=False):
        """ Re-segments a color image to isolate an object of interest using foreground masking and kmeans """
        # read params
        foreground_mask_tolerance = cfg['foreground_mask_tolerance']
        color_seg_rgb_weight = cfg['color_seg_rgb_weight']
        color_seg_num_clusters = cfg['color_seg_num_clusters']
        color_seg_hsv_weight = cfg['color_seg_hsv_weight']
        color_seg_dist_pctile = cfg['color_seg_dist_pctile']
        color_seg_dist_thresh = cfg['color_seg_dist_thresh']
        color_seg_min_bg_dist = cfg['color_seg_min_bg_dist']
        min_contour_area= cfg['min_contour_area']
        contour_dist_thresh = cfg['contour_dist_thresh']

        # foreground masking
        binary_im = color_im.foreground_mask(foreground_mask_tolerance, bgmodel=bgmodel)
        binary_im = binary_im.prune_contours(area_thresh=min_contour_area, dist_thresh=contour_dist_thresh)
        if binary_im is None:
            return None, None, None

        color_im = color_im.mask_binary(binary_im)

        # kmeans segmentation
        segment_im = color_im.segment_kmeans(color_seg_rgb_weight,
                                             color_seg_num_clusters,
                                             hue_weight=color_seg_hsv_weight)
        
        # keep the segment that is farthest from the background
        bg_dists = []
        hsv_bgmodel = 255 * np.array(colorsys.rgb_to_hsv(float(bgmodel[0]) / 255,
                                                         float(bgmodel[1]) / 255,
                                                         float(bgmodel[2]) / 255))
        hsv_bgmodel = np.r_[color_seg_rgb_weight * np.array(bgmodel), color_seg_hsv_weight * hsv_bgmodel[:1]]

        for k in range(segment_im.num_segments-1):
            seg_mask = segment_im.segment_mask(k)
            color_im_segment = color_im.mask_binary(seg_mask)
            color_im_segment_data = color_im_segment.nonzero_data()
            color_im_segment_data = np.c_[color_seg_rgb_weight * color_im_segment_data, color_seg_hsv_weight * color_im_segment.nonzero_hsv_data()[:,:1]]

            # take the median distance from the background
            bg_dist = np.median(np.linalg.norm(color_im_segment_data - hsv_bgmodel, axis=1))
            if vis_segmentation:
                logging.info('BG Dist for segment %d: %.4f' %(k, bg_dist))
            bg_dists.append(bg_dist)

        # sort by distance
        dists_and_indices = zip(np.arange(len(bg_dists)), bg_dists)
        dists_and_indices.sort(key = lambda x: x[1], reverse=True)
        
        # mask out the segment in the binary image
        if color_seg_num_clusters > 1 and abs(dists_and_indices[0][1] - dists_and_indices[1][1]) > color_seg_dist_thresh and dists_and_indices[1][1] < color_seg_min_bg_dist:
            obj_segment = dists_and_indices[0][0]
            obj_seg_mask = segment_im.segment_mask(obj_segment)
            binary_im = binary_im.mask_binary(obj_seg_mask)
            binary_im, diff_px = binary_im.center_nonzero()
            bounding_box = Box(bounding_box.min_pt.astype(np.float32) - diff_px,
                               bounding_box.max_pt.astype(np.float32) - diff_px,
                               bounding_box.frame)

        if vis_segmentation:
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(color_im.data)
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(segment_im.data)
            plt.colorbar()
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.imshow(binary_im.data, cmap=plt.cm.gray)
            plt.axis('off')
            plt.show()

        return binary_im, segment_im, bounding_box

    def detect(self, color_im, depth_im, cfg, camera_intr=None,
               T_camera_world=None,
               vis_foreground=False, vis_segmentation=False, segmask=None):
        """
        Detects all relevant objects in an rgbd image pair using foreground masking.

        Parameters
        ----------
        color_im : :obj:`ColorImage`
            color image for detection
        depth_im : :obj:`DepthImage`
            depth image for detection (corresponds to color image)
        cfg : :obj:`YamlConfig`
            parameters of detection function
        camera_intr : :obj:`CameraIntrinsics`
            intrinsics of the camera
        T_camera_world : :obj:`autolab_core.RigidTransform`
            registration of the camera to world frame
        segmask : :obj:`BinaryImage`
            optional segmask of invalid pixels

        Returns
        ------
        :obj:`list` of :obj:`RgbdDetection`
            all detections in the image
        """
        # read params
        foreground_mask_tolerance = cfg['foreground_mask_tolerance']
        min_contour_area = cfg['min_contour_area']
        max_contour_area = cfg['max_contour_area']
        min_box_area = cfg['min_box_area']
        max_box_area = cfg['max_box_area']
        box_padding_px = cfg['box_padding_px']
        crop_height = cfg['image_height']
        crop_width = cfg['image_width']
        depth_grad_thresh = cfg['depth_grad_thresh']

        w = cfg['filter_dim']

        half_crop_height = float(crop_height) / 2
        half_crop_width = float(crop_width) / 2
        half_crop_dims = np.array([half_crop_height, half_crop_width])

        fill_depth = np.max(depth_im.data[depth_im.data > 0])

        kinect2_denoising = False
        if 'kinect2_denoising' in cfg.keys() and cfg['kinect2_denoising']:
            kinect2_denoising = True
            depth_offset = cfg['kinect2_noise_offset']
            max_depth = cfg['kinect2_noise_max_depth']

        # mask image using background detection
        bgmodel = color_im.background_model()
        binary_im = color_im.foreground_mask(foreground_mask_tolerance, bgmodel=bgmodel)

        # filter the image
        y, x = np.ogrid[-w/2+1:w/2+1, -w/2+1:w/2+1]
        mask = x*x + y*y <= w/2*w/2
        filter_struct = np.zeros([w,w]).astype(np.uint8)
        filter_struct[mask] = 1
        binary_im_filtered = binary_im.apply(snm.grey_closing, structure=filter_struct)

        # find all contours
        contours = binary_im_filtered.find_contours(min_area=min_contour_area, max_area=max_contour_area)

        if vis_foreground:
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(color_im.data)
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(binary_im_filtered.data, cmap=plt.cm.gray)
            plt.axis('off')
            plt.show()

        # threshold gradients of depth
        depth_im = depth_im.threshold_gradients(depth_grad_thresh)

        # convert contours to detections
        detections = []
        for contour in contours:
            orig_box = contour.bounding_box
            if orig_box.area > min_box_area and orig_box.area < max_box_area:
                # convert orig bounding box to query bounding box
                min_pt = orig_box.center - half_crop_dims
                max_pt = orig_box.center + half_crop_dims
                query_box = Box(min_pt, max_pt, frame=orig_box.frame)

                # segment color to get refined detection
                color_thumbnail = color_im.crop(query_box.height, query_box.width, query_box.ci, query_box.cj)
                binary_thumbnail, segment_thumbnail, query_box = self._segment_color(color_thumbnail, query_box, bgmodel, cfg, vis_segmentation=vis_segmentation)
                if binary_thumbnail is None:
                    continue
            else:
                # otherwise take original bounding box
                query_box = Box(contour.bounding_box.min_pt - box_padding_px,
                                contour.bounding_box.max_pt + box_padding_px,
                                frame = contour.bounding_box.frame)

                binary_thumbnail = binary_im_filtered.crop(query_box.height, query_box.width, query_box.ci, query_box.cj)

            # crop to get thumbnails
            color_thumbnail = color_im.crop(query_box.height, query_box.width, query_box.ci, query_box.cj)
            depth_thumbnail = depth_im.crop(query_box.height, query_box.width, query_box.ci, query_box.cj)
            thumbnail_intr = camera_intr
            if camera_intr is not None:
                thumbnail_intr = camera_intr.crop(query_box.height, query_box.width, query_box.ci, query_box.cj)

            # fix depth thumbnail
            depth_thumbnail = depth_thumbnail.replace_zeros(fill_depth)
            if kinect2_denoising:
                depth_data = depth_thumbnail.data
                min_depth = np.min(depth_data)
                binary_mask_data = binary_thumbnail.data
                depth_mask_data = depth_thumbnail.mask_binary(binary_thumbnail).data
                depth_mask_data += depth_offset
                depth_data[binary_mask_data > 0] = depth_mask_data[binary_mask_data > 0]
                depth_thumbnail = DepthImage(depth_data, depth_thumbnail.frame)

            # append to detections
            detections.append(RgbdDetection(color_thumbnail,
                                            depth_thumbnail,
                                            query_box,
                                            binary_thumbnail=binary_thumbnail,
                                            contour=contour,
                                            camera_intr=thumbnail_intr))

        return detections

class PointCloudBoxDetector(RgbdDetector):
    """ Detect by removing all points in a point cloud that are outside of
    a given 3D bounding box.
    Converts all detections within a specified area into query images for a cnn.
    Optionally resegements the images using KMeans to remove spurious background pixels.
    """
    def detect(self, color_im, depth_im, cfg, camera_intr,
               T_camera_world,
               vis_foreground=False, vis_segmentation=False, segmask=None):
        """Detects all relevant objects in an rgbd image pair using foreground masking.

        Parameters
        ----------
        color_im : :obj:`ColorImage`
            color image for detection
        depth_im : :obj:`DepthImage`
            depth image for detection (corresponds to color image)
        cfg : :obj:`YamlConfig`
            parameters of detection function
        camera_intr : :obj:`CameraIntrinsics`
            intrinsics of the camera
        T_camera_world : :obj:`autolab_core.RigidTransform`
            registration of the camera to world frame
        segmask : :obj:`BinaryImage`
            optional segmask of invalid pixels

        Returns
        -------
        :obj:`list` of :obj:`RgbdDetection`
            all detections in the image
        """
        # read params
        min_pt_box = np.array(cfg['min_pt'])
        max_pt_box = np.array(cfg['max_pt'])
        min_contour_area = cfg['min_contour_area']
        max_contour_area = cfg['max_contour_area']
        min_box_area = cfg['min_box_area']
        max_box_area = cfg['max_box_area']
        box_padding_px = cfg['box_padding_px']
        crop_height = cfg['image_height']
        crop_width = cfg['image_width']
        depth_grad_thresh = cfg['depth_grad_thresh']
        point_cloud_mask_only = cfg['point_cloud_mask_only']

        w = cfg['filter_dim']

        half_crop_height = float(crop_height) / 2
        half_crop_width = float(crop_width) / 2
        half_crop_dims = np.array([half_crop_height, half_crop_width])

        fill_depth = np.max(depth_im.data[depth_im.data > 0])

        kinect2_denoising = False
        if 'kinect2_denoising' in cfg.keys() and cfg['kinect2_denoising']:
            kinect2_denoising = True
            depth_offset = cfg['kinect2_noise_offset']
            max_depth = cfg['kinect2_noise_max_depth']

        box = Box(min_pt_box, max_pt_box, 'world')

        # project into 3D
        point_cloud_cam = camera_intr.deproject(depth_im)
        point_cloud_world = T_camera_world * point_cloud_cam
        seg_point_cloud_world, _ = point_cloud_world.box_mask(box)
        seg_point_cloud_cam = T_camera_world.inverse() * seg_point_cloud_world
        depth_im_seg = camera_intr.project_to_image(seg_point_cloud_cam)

        # mask image using background detection
        bgmodel = color_im.background_model()
        binary_im = depth_im_seg.to_binary()
        if segmask is not None:
            binary_im = binary_im.mask_binary(segmask.inverse())

        # filter the image
        y, x = np.ogrid[-w/2+1:w/2+1, -w/2+1:w/2+1]
        mask = x*x + y*y <= w/2*w/2
        filter_struct = np.zeros([w,w]).astype(np.uint8)
        filter_struct[mask] = 1
        binary_im_filtered_data = snm.binary_dilation(binary_im.data, structure=filter_struct)
        binary_im_filtered = BinaryImage(binary_im_filtered_data.astype(np.uint8),
                                         frame=binary_im.frame,
                                         threshold=0)

        # find all contours
        contours = binary_im_filtered.find_contours(min_area=min_contour_area, max_area=max_contour_area)

        if vis_foreground:
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(color_im.data)
            plt.imshow(segmask.data, cmap=plt.cm.gray)
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(binary_im.data, cmap=plt.cm.gray)
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.imshow(binary_im_filtered.data, cmap=plt.cm.gray)
            plt.axis('off')
            plt.show()

        # switch to just return the mean of nonzero_px
        if point_cloud_mask_only == 1:
            center_px = np.mean(binary_im_filtered.nonzero_pixels(), axis=0)
            ci = center_px[0]
            cj = center_px[1]
            binary_thumbnail = binary_im_filtered.crop(crop_height, crop_width, ci, cj)
            color_thumbnail = color_im.crop(crop_height, crop_width, ci, cj)
            depth_thumbnail = depth_im.crop(crop_height, crop_width, ci, cj)
            thumbnail_intr = camera_intr
            if camera_intr is not None:
                thumbnail_intr = camera_intr.crop(crop_height, crop_width, ci, cj)
                
                
            query_box = Box(center_px - half_crop_dims, center_px + half_crop_dims)
            return [RgbdDetection(color_thumbnail,
                                  depth_thumbnail,
                                  query_box,
                                  binary_thumbnail=binary_thumbnail,
                                  contour=None,
                                  camera_intr=thumbnail_intr)]

        # convert contours to detections
        detections = []
        for i, contour in enumerate(contours):
            orig_box = contour.bounding_box
            logging.debug('Orig box %d area: %.3f' %(i, orig_box.area))
            if orig_box.area > min_box_area and orig_box.area < max_box_area:
                # convert orig bounding box to query bounding box
                min_pt = orig_box.center - half_crop_dims
                max_pt = orig_box.center + half_crop_dims
                query_box = Box(min_pt, max_pt, frame=orig_box.frame)

                # segment color to get refined detection
                contour_mask = binary_im_filtered.contour_mask(contour)
                binary_thumbnail = contour_mask.crop(query_box.height, query_box.width, query_box.ci, query_box.cj)

            else:
                # otherwise take original bounding box
                query_box = Box(contour.bounding_box.min_pt - box_padding_px,
                                contour.bounding_box.max_pt + box_padding_px,
                                frame = contour.bounding_box.frame)

                binary_thumbnail = binary_im_filtered.crop(query_box.height, query_box.width, query_box.ci, query_box.cj)

            # crop to get thumbnails
            color_thumbnail = color_im.crop(query_box.height, query_box.width, query_box.ci, query_box.cj)
            depth_thumbnail = depth_im.crop(query_box.height, query_box.width, query_box.ci, query_box.cj)
            thumbnail_intr = camera_intr
            if camera_intr is not None:
                thumbnail_intr = camera_intr.crop(query_box.height, query_box.width, query_box.ci, query_box.cj)

            # fix depth thumbnail
            depth_thumbnail = depth_thumbnail.replace_zeros(fill_depth)
            if kinect2_denoising:
                depth_data = depth_thumbnail.data
                min_depth = np.min(depth_data)
                binary_mask_data = binary_thumbnail.data
                depth_mask_data = depth_thumbnail.mask_binary(binary_thumbnail).data
                depth_mask_data += depth_offset
                depth_data[binary_mask_data > 0] = depth_mask_data[binary_mask_data > 0]
                depth_thumbnail = DepthImage(depth_data, depth_thumbnail.frame)

            # append to detections
            detections.append(RgbdDetection(color_thumbnail,
                                            depth_thumbnail,
                                            query_box,
                                            binary_thumbnail=binary_thumbnail,
                                            contour=contour,
                                            camera_intr=thumbnail_intr))

        return detections

class RgbdDetectorFactory:
    """ Factory class for detectors. """
    @staticmethod
    def detector(detector_type):
        """ Returns a detector of the specified type. """
        if detector_type == 'point_cloud_box':
            return PointCloudBoxDetector()
        elif detector_type == 'rgbd_foreground_mask_query':
            return RgbdForegroundMaskQueryImageDetector()
        elif detector_type == 'rgbd_foreground_mask':
            return RgbdForegroundMaskDetector()
        raise ValueError('Detector type %s not understood' %(detector_type))
