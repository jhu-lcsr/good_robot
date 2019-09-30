"""
Lean classes to encapculate images
Author: Jeff
"""
from abc import ABCMeta, abstractmethod
import IPython
import logging
import os

import six
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as PImage

import scipy.signal as ssg
import scipy.ndimage.filters as sf
import scipy.ndimage.interpolation as sni
import scipy.ndimage.morphology as snm
import scipy.spatial.distance as ssd
import scipy.signal as ssg

import sklearn.cluster as sc
import sklearn.mixture as smx
import scipy.ndimage.filters as sf
import scipy.spatial.distance as ssd
import skimage.morphology as morph
import skimage.transform as skt
import scipy.ndimage.morphology as snm

from autolab_core import PointCloud, NormalCloud, PointNormalCloud, Box, Contour
from .constants import *

BINARY_IM_MAX_VAL = np.iinfo(np.uint8).max
BINARY_IM_DEFAULT_THRESH = BINARY_IM_MAX_VAL / 2


def imresize(image, size, interp="nearest"):
    """Wrapper over `skimage.transform.resize` to mimic `scipy.misc.imresize`.

    Since `scipy.misc.imresize` has been removed in version 1.3.*, instead use
    `skimage.transform.resize`. The "lanczos" and "cubic" interpolation methods
    are not supported by `skimage.transform.resize`, however there is now
    "biquadratic", "biquartic", and "biquintic".

    Parameters
    ----------
    image : :obj:`numpy.ndarray`
        The image to resize.

    size : int, float, or tuple
        * int   - Percentage of current size.
        * float - Fraction of current size.
        * tuple - Size of the output image.

    interp : :obj:`str`, optional
        Interpolation to use for re-sizing ("neartest", "bilinear", 
        "biquadratic", "bicubic", "biquartic", "biquintic"). Default is
        "nearest".

    Returns
    -------
    :obj:`np.ndarray`
        The resized image.
    """
    skt_interp_map = {"nearest": 0, "bilinear": 1, "biquadratic": 2,
                      "bicubic": 3, "biquartic": 4, "biquintic": 5}
    if interp in ("lanczos", "cubic"):
        raise ValueError("\"lanczos\" and \"cubic\""
                         " interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation \"{}\" not"
                                      " supported.".format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size \"{}\".".format(type(size)))

    return skt.resize(image,
                      output_shape,
                      order=skt_interp_map[interp],
                      anti_aliasing=False,
                      mode="constant")

class Image(object):
    """Abstract wrapper class for images.
    """
    __metaclass__ = ABCMeta

    def __init__(self, data, frame='unspecified'):
        """Create an image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (i.e. R,G,B values). Alternatively,
            if the matrix is one dimensional, it will be interpreted as an
            N by 1 image with single element list at each pixel,
            and if the matrix is two dimensional, it
            will be a N by M matrix with a single element list at each pixel.

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError('Must initialize image with a numpy ndarray')
        if not isinstance(frame, six.string_types):
            raise ValueError('Must provide string name of frame of data')

        self._check_valid_data(data)
        self._data = self._preprocess_data(data)
        self._frame = frame
        self._encoding = 'passthrough'

    def _preprocess_data(self, data):
        """Converts a data array to the preferred 3D structure.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to process.

        Returns
        -------
        :obj:`numpy.ndarray`
            The data re-formatted (if needed) as a 3D matrix

        Raises
        ------
        ValueError
            If the data is not 1, 2, or 3D to begin with.
        """
        original_type = data.dtype
        if len(data.shape) == 1:
            data = data[:, np.newaxis, np.newaxis]
        elif len(data.shape) == 2:
            data = data[:, :, np.newaxis]
        elif len(data.shape) == 0 or len(data.shape) > 3:
            raise ValueError(
                'Illegal data array passed to image. Must be 1, 2, or 3 dimensional numpy array')
        return data.astype(original_type)

    @property
    def shape(self):
        """:obj:`tuple` of int : The shape of the data array.
        """
        return self._data.shape

    @property
    def height(self):
        """int : The number of rows in the image.
        """
        return self._data.shape[0]

    @property
    def width(self):
        """int : The number of columns in the image.
        """
        return self._data.shape[1]

    @property
    def center(self):
        """:obj:`numpy.ndarray` of int : The xy indices of the center of the
        image.
        """
        return np.array([self.height / 2, self.width / 2])

    @property
    def channels(self):
        """int : The number of channels in each pixel. For example, RGB images
        have 3 channels.
        """
        return self._data.shape[2]

    @property
    def type(self):
        """:obj:`numpy.dtype` : The data type of the image's elements.
        """
        return self._data.dtype.type

    @property
    def raw_data(self):
        """:obj:`numpy.ndarray` : The 3D array of data. The first dim is rows,
        the second is columns, and the third is pixel channels.
        """
        return self._data

    @property
    def data(self):
        """:obj:`numpy.ndarray` : The data array, but squeezed to get rid of
        extraneous dimensions.
        """
        return self._data.squeeze()

    @property
    def frame(self):
        """:obj:`str` : The frame of reference in which the image resides.
        """
        return self._frame
        """Create a new image by zeroing out data at locations not in the
        given indices.

        Parameters
        ----------
        inds : :obj:`numpy.ndarray` of int
            A 2D ndarray whose first entry is the list of row indices
            and whose second entry is the list of column indices.
            The data at these indices will not be set to zero.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type, with data not indexed by inds set
            to zero.
        """

    @property
    def encoding(self):
        """str : encoding method for the image
        """
        return self._encoding
        
    @property
    def rosmsg(self):
        """:obj:`sensor_msgs.Image` : ROS Image
        """
        from cv_bridge import CvBridge, CvBridgeError
        cv_bridge = CvBridge()
        try:
            return cv_bridge.cv2_to_imgmsg(self._data, encoding=self._encoding)
        except CvBridgeError as cv_bridge_exception:
            logging.error('%s' % (str(cv_bridge_exception)))

    @abstractmethod
    def _check_valid_data(self, data):
        """Checks that the given data is valid.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.
        """
        pass

    @abstractmethod
    def _image_data(self):
        """Returns the data in image format, with scaling and conversion to uint8 types.

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is the R/G/B entry.
        """
        pass

    @abstractmethod
    def resize(self, size, interp):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')
        """
        pass

    @staticmethod
    def can_convert(x):
        """ Returns True if x can be converted to an image, False otherwise. """
        if len(x.shape) < 2 or len(x.shape) > 3:
            return False
        dtype = x.dtype
        height = x.shape[0]
        width = x.shape[1]
        channels = 1
        if len(x.shape) == 3:
            channels = x.shape[2]
        if channels > 4:
            return False
        return True

    @staticmethod
    def from_array(x, frame='unspecified'):
        """ Converts an array of data to an Image based on the values in the array and the data format. """

        if not Image.can_convert(x):
            raise ValueError('Cannot convert array to an Image!')

        dtype = x.dtype
        height = x.shape[0]
        width = x.shape[1]
        channels = 1
        if len(x.shape) == 3:
            channels = x.shape[2]
        if dtype == np.uint8:
            if channels == 1:
                if np.any((x % BINARY_IM_MAX_VAL) > 0):
                    return GrayscaleImage(x, frame)
                return BinaryImage(x, frame)
            elif channels == 3:
                return ColorImage(x, frame)
            else:
                raise ValueError(
                    'No available image conversion for uint8 array with 2 channels')
        elif dtype == np.uint16:
            if channels != 1:
                raise ValueError(
                    'No available image conversion for uint16 array with 2 or 3 channels')
            return GrayscaleImage(x, frame)
        elif dtype == np.float32 or dtype == np.float64:
            if channels == 1:
                return DepthImage(x, frame)
            elif channels == 2:
                return GdImage(x, frame)
            elif channels == 3:
                logging.warning('Converting float array to uint8')
                return ColorImage(x.astype(np.uint8), frame)
            return RgbdImage(x, frame)
        else:
            raise ValueError(
                'Conversion for dtype %s not supported!' %
                (str(dtype)))

    def transform(self, translation, theta, method='opencv'):
        """Create a new image by translating and rotating the current image.

        Parameters
        ----------
        translation : :obj:`numpy.ndarray` of float
            The XY translation vector.
        theta : float
            Rotation angle in radians, with positive meaning counter-clockwise.
        method : :obj:`str`
            Method to use for image transformations (opencv or scipy)

        Returns
        -------
        :obj:`Image`
            An image of the same type that has been rotated and translated.
        """
        theta = np.rad2deg(theta)
        trans_map = np.float32(
            [[1, 0, translation[1]], [0, 1, translation[0]]])
        rot_map = cv2.getRotationMatrix2D(
            (self.center[1], self.center[0]), theta, 1)
        trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
        rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
        full_map = rot_map_aff.dot(trans_map_aff)
        full_map = full_map[:2, :]
        if method == 'opencv':
            im_data_tf = cv2.warpAffine(
                self.data, full_map, (self.width, self.height), flags=cv2.INTER_NEAREST)
        else:
            im_data_tf = sni.affine_transform(self.data,
                                              matrix=full_map[:, :2],
                                              offset=full_map[:, 2],
                                              order=0)
        return type(self)(
            im_data_tf.astype(
                self.data.dtype),
            frame=self._frame)

    def align(self, scale, center, angle, height, width):
        """ Create a thumbnail from the original image that
        is scaled by the given factor, centered on the center pixel, oriented along the grasp angle, and cropped to the desired height and width.

        Parameters
        ----------
        scale : float
            scale factor to apply
        center : 2D array
            array containing the row and column index of the pixel to center on
        angle : float
            angle to align the image to
        height : int
            height of the final image
        width : int
            width of the final image
        """
        # rescale
        scaled_im = self.resize(scale)

        # transform
        cx = scaled_im.center[1]
        cy = scaled_im.center[0]
        dx = cx - center[0] * scale
        dy = cy - center[1] * scale
        translation = np.array([dy, dx])
        tf_im = scaled_im.transform(translation, angle)

        # crop
        aligned_im = tf_im.crop(height, width)
        return aligned_im
        
    def gradients(self):
        """Return the gradient as a pair of numpy arrays.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of float
            The gradients of the image along each dimension.
        """
        g = np.gradient(self.data.astype(np.float32))
        return g

    def ij_to_linear(self, i, j):
        """Converts row / column coordinates to linear indices.

        Parameters
        ----------
        i : :obj:`numpy.ndarray` of int
            A list of row coordinates.

        j : :obj:`numpy.ndarray` of int
            A list of column coordinates.

        Returns
        -------
        :obj:`numpy.ndarray` of int
            A list of linear coordinates.
        """
        return i + j.dot(self.width)

    def linear_to_ij(self, linear_inds):
        """Converts linear indices to row and column coordinates.

        Parameters
        ----------
        linear_inds : :obj:`numpy.ndarray` of int
            A list of linear coordinates.

        Returns
        -------
        :obj:`numpy.ndarray` of int
            A 2D ndarray whose first entry is the list of row indices
            and whose second entry is the list of column indices.
        """
        return np.c_[linear_inds / self.width, linear_inds % self.width]

    def is_same_shape(self, other_im, check_channels=False):
        """ Checks if two images have the same height and width (and optionally channels).

        Parameters
        ----------
        other_im : :obj:`Image`
            image to compare
        check_channels : bool
            whether or not to check equality of the channels

        Returns
        -------
        bool
            True if the images are the same shape, False otherwise
        """
        if self.height == other_im.height and self.width == other_im.width:
            if check_channels and self.channels != other_im.channels:
                return False
            return True
        return False

    def mask_by_ind(self, inds):
        """Create a new image by zeroing out data at locations not in the
        given indices.

        Parameters
        ----------
        inds : :obj:`numpy.ndarray` of int
            A 2D ndarray whose first entry is the list of row indices
            and whose second entry is the list of column indices.
            The data at these indices will not be set to zero.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type, with data not indexed by inds set
            to zero.
        """
        new_data = np.zeros(self.shape)
        for ind in inds:
            new_data[ind[0], ind[1]] = self.data[ind[0], ind[1]]
        return type(self)(new_data.astype(self.data.dtype), self.frame)

    def mask_by_linear_ind(self, linear_inds):
        """Create a new image by zeroing out data at locations not in the
        given indices.

        Parameters
        ----------
        linear_inds : :obj:`numpy.ndarray` of int
            A list of linear coordinates.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type, with data not indexed by inds set
            to zero.
        """
        inds = self.linear_to_ij(linear_inds)
        return self.mask_by_ind(inds)

    def is_same_shape(self, other_im, check_channels=False):
        """Checks if two images have the same height and width
        (and optionally channels).

        Parameters
        ----------
        other_im : :obj:`Image`
            The image to compare against this one.
        check_channels : bool
            Whether or not to check equality of the channels.

        Returns
        -------
        bool
            True if the images are the same shape, False otherwise.
        """
        if self.height == other_im.height and self.width == other_im.width:
            if check_channels and self.channels != other_im.channels:
                return False
            return True
        return False

    @staticmethod
    def median_images(images):
        """Create a median Image from a list of Images.

        Parameters
        ----------
        :obj:`list` of :obj:`Image`
            A list of Image objects.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type whose data is the median of all of
            the images' data.
        """
        images_data = np.array([image.data for image in images])
        median_image_data = np.median(images_data, axis=0)

        an_image = images[0]
        return type(an_image)(
            median_image_data.astype(
                an_image.data.dtype),
            an_image.frame)

    @staticmethod
    def min_images(images):
        """Create a min Image from a list of Images.

        Parameters
        ----------
        :obj:`list` of :obj:`Image`
            A list of Image objects.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type whose data is the min of all of
            the images' data.
        """
        images_data = np.array([image.data for image in images])
        images_data[images_data == 0] = np.inf
        min_image_data = np.min(images_data, axis=0)
        min_image_data[min_image_data == np.inf] = 0.0

        an_image = images[0]
        return type(an_image)(
            min_image_data.astype(
                an_image.data.dtype),
            an_image.frame)

    def __getitem__(self, indices):
        """Index the image's data array.

        Parameters
        ----------
        indices : int or :obj:`tuple` of int
            * int - A linear index.
            * tuple - An ordered index in row, column, and (optionally) channel order.

        Returns
        -------
        item
            The indexed item.

        Raises
        ------
        ValueError
            If the index is poorly formatted or out of bounds.
        """
        # read indices
        j = None
        k = None
        if type(indices) in (tuple, np.ndarray):
            i = indices[0]
            if len(indices) > 1:
                j = indices[1]
            if len(indices) > 2:
                k = indices[2]
        else:
            i = indices

        # check indices and slicing
        if (isinstance(i, int) and i < 0) or \
           (j is not None and isinstance(j, int) and j < 0) or \
           (k is not None and isinstance(k, int) and k < 0) or \
           (isinstance(i, int) and i >= self.height) or \
           (j is not None and isinstance(j, int) and j >= self.width) or \
           (k is not None and isinstance(k, int) and k >= self.channels):
            raise ValueError('Out of bounds indexing')
        if (isinstance(i, slice) and i.start < 0) or \
           (j is not None and isinstance(j, slice) and j.start < 0) or \
           (k is not None and isinstance(k, slice) and k.start < 0) or \
           (isinstance(i, slice) and i.stop > self.height) or \
           (j is not None and isinstance(j, slice) and j.stop > self.width) or \
           (k is not None and isinstance(k, slice) and k.stop > self.channels):
            raise ValueError('Out of bounds slicing')
        if k is not None and isinstance(
                k, int) and k > 1 and self.channels < 3:
            raise ValueError('Illegal indexing. Image is not 3 dimensional')

        # linear indexing
        if j is None:
            return self._data[i]
        # return the channel vals for the i, j pixel
        if k is None:
            return self._data[i, j, :]
        return self._data[i, j, k]

    def apply(self, method, *args, **kwargs):
        """Create a new image by applying a function to this image's data.

        Parameters
        ----------
        method : :obj:`function`
            A function to call on the data. This takes in a ndarray
            as its first argument and optionally takes other arguments.
            It should return a modified data ndarray.

        args : arguments
            Additional args for method.

        kwargs : keyword arguments
            Additional keyword arguments for method.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type with new data generated by calling
            method on the current image's data.
        """
        data = method(self.data, *args, **kwargs)
        return type(self)(data.astype(self.type), self.frame)

    def copy(self):
        """ Returns a copy of this image.

        Returns
        -------
        :obj:`Image`
            copy of this image
        """
        return type(self)(self.data.copy(), self.frame)

    def crop(self, height, width, center_i=None, center_j=None):
        """Crop the image centered around center_i, center_j.

        Parameters
        ----------
        height : int
            The height of the desired image.

        width : int
            The width of the desired image.

        center_i : int
            The center height point at which to crop. If not specified, the center
            of the image is used.

        center_j : int
            The center width point at which to crop. If not specified, the center
            of the image is used.

        Returns
        -------
        :obj:`Image`
            A cropped Image of the same type.
        """
        # compute crop center px
        height = int(np.round(height))
        width = int(np.round(width))
        if center_i is None:
            center_i = float(self.height) / 2
        if center_j is None:
            center_j = float(self.width) / 2

        # crop using PIL
        desired_start_row = int(np.floor(center_i - float(height) / 2))
        desired_end_row = int(np.floor(center_i + float(height) / 2))
        desired_start_col = int(np.floor(center_j - float(width) / 2))
        desired_end_col = int(np.floor(center_j + float(width) / 2))

        pil_im = PImage.fromarray(self.data)
        cropped_pil_im = pil_im.crop((desired_start_col,
                                      desired_start_row,
                                      desired_end_col,
                                      desired_end_row))
        crop_data = np.array(cropped_pil_im)

        if crop_data.shape[0] != height or crop_data.shape[1] != width:
            raise ValueError('Crop dims are incorrect')

        return type(self)(crop_data.astype(self.data.dtype), self._frame)

    def focus(self, height, width, center_i=None, center_j=None):
        """Zero out all of the image outside of a crop box.

        Parameters
        ----------
        height : int
            The height of the desired crop box.

        width : int
            The width of the desired crop box.

        center_i : int
            The center height point of the crop box. If not specified, the center
            of the image is used.

        center_j : int
            The center width point of the crop box. If not specified, the center
            of the image is used.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type and size that is zeroed out except
            within the crop box.
        """
        if center_i is None:
            center_i = self.height / 2
        if center_j is None:
            center_j = self.width / 2

        start_row = int(max(0, center_i - height / 2))
        end_row = int(min(self.height - 1, center_i + height / 2))
        start_col = int(max(0, center_j - width / 2))
        end_col = int(min(self.width - 1, center_j + width / 2))

        focus_data = np.zeros(self._data.shape)
        focus_data[start_row:end_row + 1, start_col:end_col + \
            1] = self._data[start_row:end_row + 1, start_col:end_col + 1]
        return type(self)(focus_data.astype(self._data.dtype), self._frame)

    def center_nonzero(self):
        """Recenters the image on the mean of the coordinates of nonzero pixels.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type and size that is re-centered
            at the mean location of the non-zero pixels.
        """
        # get the center of the nonzero pixels
        nonzero_px = np.where(self._data != 0.0)
        nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
        mean_px = np.mean(nonzero_px, axis=0)
        center_px = (np.array(self.shape) / 2.0)[:2]
        diff_px = center_px - mean_px

        # transform image
        nonzero_px_tf = nonzero_px + diff_px
        nonzero_px_tf[:, 0] = np.max(
            np.c_[np.zeros(nonzero_px_tf[:, 0].shape), nonzero_px_tf[:, 0]], axis=1)
        nonzero_px_tf[:, 0] = np.min(np.c_[(
            self.height - 1) * np.ones(nonzero_px_tf[:, 0].shape), nonzero_px_tf[:, 0]], axis=1)
        nonzero_px_tf[:, 1] = np.max(
            np.c_[np.zeros(nonzero_px_tf[:, 1].shape), nonzero_px_tf[:, 1]], axis=1)
        nonzero_px_tf[:, 1] = np.min(np.c_[(
            self.width - 1) * np.ones(nonzero_px_tf[:, 1].shape), nonzero_px_tf[:, 1]], axis=1)
        nonzero_px = nonzero_px.astype(np.uint16)
        nonzero_px_tf = nonzero_px_tf.astype(np.uint16)
        shifted_data = np.zeros(self.shape)
        shifted_data[nonzero_px_tf[:, 0], nonzero_px_tf[:, 1],
                     :] = self.data[nonzero_px[:, 0], nonzero_px[:, 1]].reshape(-1, self.channels)

        return type(self)(
            shifted_data.astype(
                self.data.dtype), frame=self._frame), diff_px

    def nonzero_pixels(self):
        """ Return an array of the nonzero pixels.

        Returns
        -------
        :obj:`numpy.ndarray`
             Nx2 array of the nonzero pixels
        """
        nonzero_px = np.where(np.sum(self.raw_data, axis=2) > 0)
        nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
        return nonzero_px

    def zero_pixels(self):
        """ Return an array of the zero pixels.

        Returns
        -------
        :obj:`numpy.ndarray`
             Nx2 array of the zero pixels
        """
        zero_px = np.where(np.sum(self.raw_data, axis=2) == 0)
        zero_px = np.c_[zero_px[0], zero_px[1]]
        return zero_px

    def nan_pixels(self):
        """ Return an array of the NaN pixels.

        Returns
        -------
        :obj:`numpy.ndarray`
             Nx2 array of the NaN pixels
        """
        nan_px = np.where(np.isnan(np.sum(self.raw_data, axis=2)))
        nan_px = np.c_[nan_px[0], nan_px[1]]
        return nan_px

    def finite_pixels(self):
        """ Return an array of the finite pixels.

        Returns
        -------
        :obj:`numpy.ndarray`
             Nx2 array of the finite pixels
        """
        finite_px = np.where(np.isfinite(self.data))
        finite_px = np.c_[finite_px[0], finite_px[1]]
        return finite_px

    def nonzero_data(self):
        """ Returns the values in the image at the nonzero pixels

        Returns
        -------
        :obj:`numpy.ndarray`
             NxC array of the nonzero data
        """
        nonzero_px = self.nonzero_pixels()
        return self.data[nonzero_px[:, 0], nonzero_px[:, 1], ...]

    def replace_zeros(self, val, zero_thresh=0.0):
        """ Replaces all zeros in the image with a specified value

        Returns
        -------
        image dtype
             value to replace zeros with
        """
        new_data = self.data.copy()
        new_data[new_data <= zero_thresh] = val
        return type(self)(new_data.astype(self.data.dtype), frame=self._frame)

    def save(self, filename):
        """Writes the image to a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to save the image to. Must be one of .png, .jpg,
            .npy, or .npz.

        Raises
        ------
        ValueError
            If an unsupported file type is specified.
        """
        filename = str(filename)
        file_root, file_ext = os.path.splitext(filename)
        if file_ext in COLOR_IMAGE_EXTS:
            im_data = self._image_data()
            if im_data.dtype.type == np.uint8:
                pil_image = PImage.fromarray(im_data.squeeze())
                pil_image.save(filename)
            else:
                try:
                    import png
                except:
                    raise ValueError('PyPNG not installed! Cannot save 16-bit images')
                png.fromarray(im_data, 'L').save(filename)
        elif file_ext == '.npy':
            np.save(filename, self._data)
        elif file_ext == '.npz':
            np.savez_compressed(filename, self._data)
        else:
            raise ValueError('Extension %s not supported' % (file_ext))

    def savefig(self, output_path, title, dpi=400, format='png', cmap=None):
        """Write the image to a file using pyplot.

        Parameters
        ----------
        output_path : :obj:`str`
            The directory in which to place the file.

        title : :obj:`str`
            The title of the file in which to save the image.

        dpi : int
            The resolution in dots per inch.

        format : :obj:`str`
            The file format to save. Available options include .png, .pdf, .ps,
            .eps, and .svg.

        cmap : :obj:`Colormap`, optional
            A Colormap object fo the pyplot.
        """
        plt.figure()
        plt.imshow(self.data, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        title_underscore = title.replace(' ', '_')
        plt.savefig(
            os.path.join(
                output_path,
                '{0}.{1}'.format(
                    title_underscore,
                    format)),
            dpi=dpi,
            format=format)

    @staticmethod
    def load_data(filename):
        """Loads a data matrix from a given file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        Returns
        -------
        :obj:`numpy.ndarray`
            The data array read from the file.
        """
        file_root, file_ext = os.path.splitext(filename)
        data = None
        if file_ext.lower() in COLOR_IMAGE_EXTS:
            data = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        elif file_ext == '.npy':
            data = np.load(filename)
        elif file_ext == '.npz':
            data = np.load(filename)['arr_0']
        else:
            raise ValueError('Extension %s not supported' % (file_ext))
        return data


class ColorImage(Image):
    """An RGB color image.
    """

    def __init__(self, data, frame='unspecified', encoding='rgb8'):
        """Create a color image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (i.e. R,G,B values). Alternatively, the
            image may have a single channel, in which case it is interpreted as
            greyscale.

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        encoding : :obj:`str`
            Either rgb8 or bgr8, depending on the channel storage mode

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        Image.__init__(self, data, frame)
        self._encoding = encoding
        if self._encoding != 'rgb8' and self._encoding != 'bgr8':
            raise ValueError(
                'Illegal encoding: %s. Please use rgb8 or bgr8' %
                (self._encoding))
        if self._encoding == 'rgb8':
            self.r_axis = 0
            self.g_axis = 1
            self.b_axis = 2
        else:
            self.r_axis = 2
            self.g_axis = 1
            self.b_axis = 0

    def _check_valid_data(self, data):
        """Checks that the given data is a uint8 array with one or three
        channels.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.uint8:
            raise ValueError(
                'Illegal data type. Color images only support uint8 arrays')

        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError(
                'Illegal data type. Color images only support three channels')

    def _image_data(self):
        """Returns the data in image format, with scaling and conversion to uint8 types.

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is the R/G/B entry.
        """
        return self._data

    @property
    def r_data(self):
        """:obj:`numpy.ndarray` of uint8 : The red-channel data.
        """
        return self.data[:, :, self.r_axis]

    @property
    def g_data(self):
        """:obj:`numpy.ndarray` of uint8 : The green-channel data.
        """
        return self.data[:, :, self.g_axis]

    @property
    def b_data(self):
        """:obj:`numpy.ndarray` of uint8 : The blue-channel data.
        """
        return self.data[:, :, self.b_axis]

    def bgr2rgb(self):
        """ Converts data using the cv conversion. """
        new_data = cv2.cvtColor(self.raw_data, cv2.COLOR_BGR2RGB)
        return ColorImage(new_data, frame=self.frame, encoding='rgb8')

    def rgb2bgr(self):
        """ Converts data using the cv conversion. """
        new_data = cv2.cvtColor(self.raw_data, cv2.COLOR_RGB2BGR)
        return ColorImage(new_data, frame=self.frame, encoding='bgr8')

    def swap_channels(self, channel_swap):
        """ Swaps the two channels specified in the tuple.

        Parameters
        ----------
        channel_swap : :obj:`tuple` of int
            the two channels to swap

        Returns
        -------
        :obj:`ColorImage`
            color image with cols swapped
        """
        if len(channel_swap) != 2:
            raise ValueError('Illegal value for channel swap')
        ci = channel_swap[0]
        cj = channel_swap[1]
        if ci < 0 or ci > 2 or cj < 0 or cj > 2:
            raise ValueError('Channels must be between 0 and 1')
        new_data = self.data.copy()
        new_data[:, :, ci] = self.data[:, :, cj]
        new_data[:, :, cj] = self.data[:, :, ci]
        return ColorImage(new_data, frame=self._frame)

    def resize(self, size, interp='bilinear'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')

        Returns
        -------
        :obj:`ColorImage`
            The resized image.
        """
        resized_data = imresize(self.data, size, interp=interp).astype(np.uint8)
        return ColorImage(resized_data, self._frame)

    def find_chessboard(self, sx=6, sy=9):
        """Finds the corners of an sx X sy chessboard in the image.

        Parameters
        ----------
        sx : int
            Number of chessboard corners in x-direction.
        sy : int
            Number of chessboard corners in y-direction.

        Returns
        -------
        :obj:`list` of :obj:`numpy.ndarray`
            A list containing the 2D points of the corners of the detected
            chessboard, or None if no chessboard found.
        """
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((sx * sy, 3), np.float32)
        objp[:, :2] = np.mgrid[0:sx, 0:sy].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        # create images
        img = self.data.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (sx, sy), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            if corners is not None:
                return corners.squeeze()
        return None

    def mask_binary(self, binary_im):
        """Create a new image by zeroing out data at locations
        where binary_im == 0.0.

        Parameters
        ----------
        binary_im : :obj:`BinaryImage`
            A BinaryImage of the same size as this image, with pixel values of either
            zero or one. Wherever this image has zero pixels, we'll zero out the
            pixels of the new image.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type, masked by the given binary image.
        """
        data = np.copy(self._data)
        ind = np.where(binary_im.data == 0)
        data[ind[0], ind[1], :] = 0.0
        return ColorImage(data, self._frame)

    def foreground_mask(
            self,
            tolerance,
            ignore_black=True,
            use_hsv=False,
            scale=8,
            bgmodel=None):
        """Creates a binary image mask for the foreground of an image against
        a uniformly colored background. The background is assumed to be the mode value of the histogram
        for each of the color channels.

        Parameters
        ----------
        tolerance : int
            A +/- level from the detected mean backgroud color. Pixels withing
            this range will be classified as background pixels and masked out.

        ignore_black : bool
            If True, the zero pixels will be ignored
            when computing the background model.

        use_hsv : bool
            If True, image will be converted to HSV for background model
            generation.

        scale : int
            Size of background histogram bins -- there will be BINARY_IM_MAX_VAL/size bins
            in the color histogram for each channel.

        bgmodel : :obj:`list` of int
            A list containing the red, green, and blue channel modes of the
            background. If this is None, a background model will be generated
            using the other parameters.

        Returns
        -------
        :obj:`BinaryImage`
            A binary image that masks out the background from the current
            ColorImage.
        """
        # get a background model
        if bgmodel is None:
            bgmodel = self.background_model(ignore_black=ignore_black,
                                            use_hsv=use_hsv,
                                            scale=scale)

        # get the bounds
        lower_bound = np.array(
            [bgmodel[i] - tolerance for i in range(self.channels)])
        upper_bound = np.array(
            [bgmodel[i] + tolerance for i in range(self.channels)])
        orig_zero_indices = np.where(np.sum(self._data, axis=2) == 0)

        # threshold
        binary_data = cv2.inRange(self.data, lower_bound, upper_bound)
        binary_data[:, :, ] = (BINARY_IM_MAX_VAL - binary_data[:, :, ])
        binary_data[orig_zero_indices[0], orig_zero_indices[1], ] = 0.0
        binary_im = BinaryImage(binary_data.astype(np.uint8), frame=self.frame)
        return binary_im

    def background_model(self, ignore_black=True, use_hsv=False, scale=8):
        """Creates a background model for the given image. The background
        color is given by the modes of each channel's histogram.

        Parameters
        ----------
        ignore_black : bool
            If True, the zero pixels will be ignored
            when computing the background model.

        use_hsv : bool
            If True, image will be converted to HSV for background model
            generation.

        scale : int
            Size of background histogram bins -- there will be BINARY_IM_MAX_VAL/size bins
            in the color histogram for each channel.

        Returns
        -------
            A list containing the red, green, and blue channel modes of the
            background.
        """
        # hsv color
        data = self.data
        if use_hsv:
            pil_im = PImage.fromarray(self._data)
            pil_im = pil_im.convert('HSV')
            data = np.asarray(pil_im)

        # find the black pixels
        nonblack_pixels = np.where(np.sum(self.data, axis=2) > 0)
        r_data = self.r_data
        g_data = self.g_data
        b_data = self.b_data
        if ignore_black:
            r_data = r_data[nonblack_pixels[0], nonblack_pixels[1]]
            g_data = g_data[nonblack_pixels[0], nonblack_pixels[1]]
            b_data = b_data[nonblack_pixels[0], nonblack_pixels[1]]

        # generate histograms for each channel
        bounds = (0, np.iinfo(np.uint8).max + 1)
        num_bins = bounds[1] / scale
        r_hist, _ = np.histogram(r_data, bins=num_bins, range=bounds)
        g_hist, _ = np.histogram(g_data, bins=num_bins, range=bounds)
        b_hist, _ = np.histogram(b_data, bins=num_bins, range=bounds)
        hists = (r_hist, g_hist, b_hist)

        # find the thesholds as the modes of the image
        modes = [0 for i in range(self.channels)]
        for i in range(self.channels):
            modes[i] = scale * np.argmax(hists[i])

        return modes

    def draw_box(self, box):
        """Draw a white box on the image.

        Parameters
        ----------
        :obj:`autolab_core.Box`
            A 2D box to draw in the image.

        Returns
        -------
        :obj:`ColorImage`
            A new image that is the same as the current one, but with
            the white box drawn in.
        """
        box_data = self._data.copy()
        min_i = box.min_pt[1]
        min_j = box.min_pt[0]
        max_i = box.max_pt[1]
        max_j = box.max_pt[0]

        # draw the vertical lines
        for j in range(min_j, max_j):
            box_data[min_i, j, :] = BINARY_IM_MAX_VAL * np.ones(self.channels)
            box_data[max_i, j, :] = BINARY_IM_MAX_VAL * np.ones(self.channels)

        # draw the horizontal lines
        for i in range(min_i, max_i):
            box_data[i, min_j, :] = BINARY_IM_MAX_VAL * np.ones(self.channels)
            box_data[i, max_j, :] = BINARY_IM_MAX_VAL * np.ones(self.channels)

        return ColorImage(box_data, self._frame)

    def nonzero_hsv_data(self):
        """ Computes non zero hsv values.

        Returns
        -------
        :obj:`numpy.ndarray`
            array of the hsv values for the image
        """
        hsv_data = cv2.cvtColor(self.data, cv2.COLOR_BGR2HSV)
        nonzero_px = self.nonzero_pixels()
        return hsv_data[nonzero_px[:, 0], nonzero_px[:, 1], ...]

    def segment_kmeans(self, rgb_weight, num_clusters, hue_weight=0.0):
        """
        Segment a color image using KMeans based on spatial and color distances.
        Black pixels will automatically be assigned to their own 'background' cluster.

        Parameters
        ----------
        rgb_weight : float
            weighting of RGB distance relative to spatial and hue distance
        num_clusters : int
            number of clusters to use
        hue_weight : float
            weighting of hue from hsv relative to spatial and RGB distance

        Returns
        -------
        :obj:`SegmentationImage`
            image containing the segment labels
        """
        # form features array
        label_offset = 1
        nonzero_px = np.where(self.data != 0.0)
        nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]

        # get hsv data if specified
        color_vals = rgb_weight * \
            self._data[nonzero_px[:, 0], nonzero_px[:, 1], :]
        if hue_weight > 0.0:
            hsv_data = cv2.cvtColor(self.data, cv2.COLOR_BGR2HSV)
            color_vals = np.c_[color_vals, hue_weight *
                               hsv_data[nonzero_px[:, 0], nonzero_px[:, 1], :1]]
        features = np.c_[nonzero_px, color_vals.astype(np.float32)]

        # perform KMeans clustering
        kmeans = sc.KMeans(n_clusters=num_clusters)
        labels = kmeans.fit_predict(features)

        # create output label array
        label_im = np.zeros([self.height, self.width]).astype(np.uint8)
        label_im[nonzero_px[:, 0], nonzero_px[:, 1]] = labels + label_offset
        return SegmentationImage(label_im, frame=self.frame)

    def inpaint(self, win_size=3, rescale_factor=1.0):
        """ Fills in the zero pixels in the image.

        Parameters
        ----------
        win_size : int
            size of window to use for inpainting
        rescale_factor : float
            amount to rescale the image for inpainting, smaller numbers increase speed

        Returns
        -------
        :obj:`ColorImage`
            color image with zero pixels filled in
        """
        # get original shape
        orig_shape = (self.height, self.width)
        
        # resize the image
        resized_data = self.resize(rescale_factor, interp='nearest').data

        # inpaint smaller image
        mask = 1 * (np.sum(resized_data, axis=2) == 0)
        inpainted_data = cv2.inpaint(resized_data, mask.astype(np.uint8),
                                     win_size, cv2.INPAINT_TELEA)
        inpainted_im = ColorImage(inpainted_data, frame=self.frame)

        # fill in zero pixels with inpainted and resized image
        filled_data = inpainted_im.resize(
            orig_shape, interp='bilinear').data
        new_data = self.data
        new_data[self.data == 0] = filled_data[self.data == 0]
        return ColorImage(new_data, frame=self.frame)

    def to_binary(self, threshold=0.0):
        """Converts the color image to binary.

        Returns
        -------
        :obj:`BinaryImage`
            Binary image corresponding to the nonzero px of the original image
        """
        data = BINARY_IM_MAX_VAL * (self._data > threshold)
        return BinaryImage(data[:, :, 0].astype(np.uint8), self._frame)

    def to_grayscale(self):
        """Converts the color image to grayscale using OpenCV.

        Returns
        -------
        :obj:`GrayscaleImage`
            Grayscale image corresponding to original color image.
        """
        gray_data = cv2.cvtColor(self.data, cv2.COLOR_RGB2GRAY)
        return GrayscaleImage(gray_data, frame=self.frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Creates a ColorImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`ColorImage`
            The new color image.
        """
        data = Image.load_data(filename).astype(np.uint8)
        return ColorImage(data, frame)


class DepthImage(Image):
    """A depth image in which individual pixels have a single floating-point
    depth channel.
    """

    def __init__(self, data, frame='unspecified'):
        """Create a depth image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (depths as floating point numbers).

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        Image.__init__(self, data, frame)
        self._data = self._data.astype(np.float32)
        self._data[np.isnan(self._data)] = 0.0
        self._encoding = 'passthrough'

    def _check_valid_data(self, data):
        """Checks that the given data is a float array with one channel.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.float32 and \
                data.dtype.type is not np.float64:
            raise ValueError(
                'Illegal data type. Depth images only support float arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError(
                'Illegal data type. Depth images only support single channel')

    def _image_data(self, normalize=False,
                    min_depth=MIN_DEPTH,
                    max_depth=MAX_DEPTH,
                    twobyte=False):
        """Returns the data in image format, with scaling and conversion to uint8 types.

        Parameters
        ----------
        normalize : bool
            whether or not to normalize by the min and max depth of the image
        min_depth : float
            minimum depth value for the normalization
        max_depth : float
            maximum depth value for the normalization
        twobyte: bool
            whether or not to use 16-bit encoding

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is a set of 3 RGB values, each of
            which is simply the depth entry scaled to between 0 and BINARY_IM_MAX_VAL.
        """
        max_val = BINARY_IM_MAX_VAL
        if twobyte:
            max_val = np.iinfo(np.uint16).max
        
        if normalize:
            min_depth = np.min(self._data)
            max_depth = np.max(self._data)
            depth_data = (self._data - min_depth) / (max_depth - min_depth)
            depth_data = float(max_val) * depth_data.squeeze()
        else:
            zero_px = np.where(self._data == 0)
            zero_px = np.c_[zero_px[0], zero_px[1], zero_px[2]]
            depth_data = ((self._data - min_depth) * \
                          (float(max_val) / (max_depth - min_depth))).squeeze()
            depth_data[zero_px[:,0], zero_px[:,1]] = 0
        im_data = np.zeros([self.height, self.width, 3])
        im_data[:, :, 0] = depth_data
        im_data[:, :, 1] = depth_data
        im_data[:, :, 2] = depth_data
        if twobyte:
            return im_data.astype(np.uint16)
        return im_data.astype(np.uint8)

    def save(self, filename, normalize=False, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, twobyte=False):
        """Writes the image to a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to save the image to. Must be one of .png, .jpg,
            .npy, or .npz.

        Raises
        ------
        ValueError
            If an unsupported file type is specified.
        """
        filename = str(filename)
        file_root, file_ext = os.path.splitext(filename)
        if file_ext in COLOR_IMAGE_EXTS:
            im_data = self._image_data(normalize=normalize, min_depth=min_depth, max_depth=max_depth, twobyte=twobyte)
            if im_data.dtype.type == np.uint8:
                pil_image = PImage.fromarray(im_data.squeeze())
                pil_image.save(filename)
            else:
                try:
                    import png
                except:
                    raise ValueError('PyPNG not installed! Cannot save 16-bit images')
                png.fromarray(im_data, 'L').save(filename)
        else:
            super().save(filename)

    def resize(self, size, interp='bilinear'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')

        Returns
        -------
        :obj:`DepthImage`
            The resized image.
        """
        resized_data = imresize(self.data, size, interp=interp)
        return DepthImage(resized_data, self._frame)

    def threshold(self, front_thresh=0.0, rear_thresh=100.0):
        """Creates a new DepthImage by setting all depths less than
        front_thresh and greater than rear_thresh to 0.

        Parameters
        ----------
        front_thresh : float
            The lower-bound threshold.

        rear_thresh : float
            The upper bound threshold.

        Returns
        -------
        :obj:`DepthImage`
            A new DepthImage created from the thresholding operation.
        """
        data = np.copy(self._data)
        data[data < front_thresh] = 0.0
        data[data > rear_thresh] = 0.0
        return DepthImage(data, self._frame)

    def threshold_gradients(self, grad_thresh):
        """Creates a new DepthImage by zeroing out all depths
        where the magnitude of the gradient at that point is
        greater than grad_thresh.

        Parameters
        ----------
        grad_thresh : float
            A threshold for the gradient magnitude.

        Returns
        -------
        :obj:`DepthImage`
            A new DepthImage created from the thresholding operation.
        """
        data = np.copy(self._data)
        gx, gy = self.gradients()
        gradients = np.zeros([gx.shape[0], gx.shape[1], 2])
        gradients[:, :, 0] = gx
        gradients[:, :, 1] = gy
        gradient_mags = np.linalg.norm(gradients, axis=2)
        ind = np.where(gradient_mags > grad_thresh)
        data[ind[0], ind[1]] = 0.0
        return DepthImage(data, self._frame)

    def threshold_gradients_pctile(self, thresh_pctile, min_mag=0.0):
        """Creates a new DepthImage by zeroing out all depths
        where the magnitude of the gradient at that point is
        greater than some percentile of all gradients.

        Parameters
        ----------
        thresh_pctile : float
            percentile to threshold all gradients above
        min_mag : float
            minimum magnitude of the gradient

        Returns
        -------
        :obj:`DepthImage`
            A new DepthImage created from the thresholding operation.
        """
        data = np.copy(self._data)
        gx, gy = self.gradients()
        gradients = np.zeros([gx.shape[0], gx.shape[1], 2])
        gradients[:, :, 0] = gx
        gradients[:, :, 1] = gy
        gradient_mags = np.linalg.norm(gradients, axis=2)
        grad_thresh = np.percentile(gradient_mags, thresh_pctile)
        ind = np.where(
            (gradient_mags > grad_thresh) & (
                gradient_mags > min_mag))
        data[ind[0], ind[1]] = 0.0
        return DepthImage(data, self._frame)

    def inpaint(self, rescale_factor=1.0):
        """ Fills in the zero pixels in the image.

        Parameters
        ----------
        rescale_factor : float
            amount to rescale the image for inpainting, smaller numbers increase speed

        Returns
        -------
        :obj:`DepthImage`
            depth image with zero pixels filled in
        """
        # get original shape
        orig_shape = (self.height, self.width)

        # form inpaint kernel
        inpaint_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # resize the image
        resized_data = self.resize(rescale_factor, interp='nearest').data

        # inpaint the smaller image
        cur_data = resized_data.copy()
        zeros = (cur_data == 0)
        while np.any(zeros):
            neighbors = ssg.convolve2d((cur_data != 0), inpaint_kernel,
                                       mode='same', boundary='symm')
            avg_depth = ssg.convolve2d(cur_data, inpaint_kernel,
                                       mode='same', boundary='symm')
            avg_depth[neighbors > 0] = avg_depth[neighbors > 0] / \
                neighbors[neighbors > 0]
            avg_depth[neighbors == 0] = 0
            avg_depth[resized_data > 0] = resized_data[resized_data > 0]
            cur_data = avg_depth
            zeros = (cur_data == 0)

        # fill in zero pixels with inpainted and resized image
        inpainted_im = DepthImage(cur_data, frame=self.frame)
        filled_data = inpainted_im.resize(
            orig_shape, interp='bilinear').data
        new_data = np.copy(self.data)
        new_data[self.data == 0] = filled_data[self.data == 0]
        return DepthImage(new_data, frame=self.frame)

    def invalid_pixel_mask(self):
        """ Returns a binary mask for the NaN- and zero-valued pixels.
        Serves as a mask for invalid pixels.

        Returns
        -------
        :obj:`BinaryImage`
            Binary image where a pixel value greater than zero indicates an invalid pixel.
        """
        # init mask buffer
        mask = np.zeros([self.height, self.width, 1]).astype(np.uint8)

        # update invalid pixels
        zero_pixels = self.zero_pixels()
        nan_pixels = self.nan_pixels()
        mask[zero_pixels[:, 0], zero_pixels[:, 1]] = BINARY_IM_MAX_VAL
        mask[nan_pixels[:, 0], nan_pixels[:, 1]] = BINARY_IM_MAX_VAL
        return BinaryImage(mask, frame=self.frame)

    def mask_binary(self, binary_im):
        """Create a new image by zeroing out data at locations
        where binary_im == 0.0.

        Parameters
        ----------
        binary_im : :obj:`BinaryImage`
            A BinaryImage of the same size as this image, with pixel values of either
            zero or one. Wherever this image has zero pixels, we'll zero out the
            pixels of the new image.

        Returns
        -------
        :obj:`DepthImage`
            A new DepthImage of the same type, masked by the given binary image.
        """
        data = np.copy(self._data)
        ind = np.where(binary_im.data == 0)
        data[ind[0], ind[1]] = 0.0
        return DepthImage(data, self._frame)

    def pixels_farther_than(self, depth_im, filter_equal_depth=False):
        """
        Returns the pixels that are farther away
        than those in the corresponding depth image.

        Parameters
        ----------
        depth_im : :obj:`DepthImage`
            depth image to query replacement with
        filter_equal_depth : bool
            whether or not to mark depth values that are equal

        Returns
        -------
        :obj:`numpy.ndarray`
            the pixels
        """
        # take closest pixel
        if filter_equal_depth:
            farther_px = np.where((self.data > depth_im.data) & (np.isfinite(depth_im.data)))
        else:
            farther_px = np.where((self.data >= depth_im.data) & (np.isfinite(depth_im.data)))
        farther_px = np.c_[farther_px[0], farther_px[1]]
        return farther_px

    def combine_with(self, depth_im):
        """
        Replaces all zeros in the source depth image with the value of a different depth image

        Parameters
        ----------
        depth_im : :obj:`DepthImage`
            depth image to combine with

        Returns
        -------
        :obj:`DepthImage`
            the combined depth image
        """
        new_data = self.data.copy()
        # replace zero pixels
        new_data[new_data == 0] = depth_im.data[new_data == 0]
        # take closest pixel
        new_data[(new_data > depth_im.data) & (depth_im.data > 0)] = depth_im.data[(
            new_data > depth_im.data) & (depth_im.data > 0)]
        return DepthImage(new_data, frame=self.frame)

    def to_binary(self, threshold=0.0):
        """Creates a BinaryImage from the depth image. Points where the depth
        is greater than threshold are converted to ones, and all other points
        are zeros.

        Parameters
        ----------
        threshold : float
            The depth threshold.

        Returns
        -------
        :obj:`BinaryImage`
            A BinaryImage where all 1 points had a depth greater than threshold
            in the DepthImage.
        """
        data = BINARY_IM_MAX_VAL * (self._data > threshold)
        return BinaryImage(data.astype(np.uint8), self._frame)

    def to_color(self, normalize=False):
        """ Convert to a color image.

        Parameters
        ----------
        normalize : bool
             whether or not to normalize by the maximum depth

        Returns
        -------
        :obj:`ColorImage`
            color image corresponding to the depth image
        """
        im_data = self._image_data(normalize=normalize)
        return ColorImage(im_data, frame=self._frame)

    def to_float(self):
        """ Converts to 32-bit data.

        Returns
        -------
        :obj:`DepthImage`
            depth image with 32 bit float data
        """
        return DepthImage(self.data.astype(np.float32), frame=self.frame)

    def point_normal_cloud(self, camera_intr):
        """Computes a PointNormalCloud from the depth image.

        Parameters
        ----------
        camera_intr : :obj:`CameraIntrinsics`
            The camera parameters on which this depth image was taken.

        Returns
        -------
        :obj:`autolab_core.PointNormalCloud`
            A PointNormalCloud created from the depth image.
        """
        point_cloud_im = camera_intr.deproject_to_image(self)
        normal_cloud_im = point_cloud_im.normal_cloud_im()
        point_cloud = point_cloud_im.to_point_cloud()
        normal_cloud = normal_cloud_im.to_normal_cloud()
        return PointNormalCloud(
            point_cloud.data,
            normal_cloud.data,
            frame=self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Creates a DepthImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`DepthImage`
            The new depth image.
        """
        file_root, file_ext = os.path.splitext(filename)
        data = Image.load_data(filename)
        if file_ext.lower() in COLOR_IMAGE_EXTS:
            data = (data * (MAX_DEPTH / BINARY_IM_MAX_VAL)).astype(np.float32)
        return DepthImage(data, frame)


class IrImage(Image):
    """An IR image in which individual pixels have a single uint16 channel.
    """

    def __init__(self, data, frame='unspecified'):
        """Create an IR image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (IR values as uint16's).

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the given data is a uint16 array with one channel.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.uint16:
            raise ValueError(
                'Illegal data type. IR images only support 16-bit uint arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError(
                'Illegal data type. IR images only support single channel ')

    def _image_data(self):
        """Returns the data in image format, with scaling and conversion to uint8 types.

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is simply the IR entry scaled to between 0 and BINARY_IM_MAX_VAL.
        """
        return (self._data * (float(BINARY_IM_MAX_VAL) / MAX_IR)).astype(np.uint8)

    def resize(self, size, interp='bilinear'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')

        Returns
        -------
        :obj:`IrImage`
            The resized image.
        """
        resized_data = imresize(self._data, size, interp=interp).astype(np.uint16)
        return IrImage(resized_data, self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Creates an IrImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`IrImage`
            The new IR image.
        """
        data = Image.load_data(filename)
        data = (data * (MAX_IR / BINARY_IM_MAX_VAL)).astype(np.uint16)
        return IrImage(data, frame)


class GrayscaleImage(Image):
    """A grayscale image in which individual pixels have a single uint8 channel.
    """

    def __init__(self, data, frame='unspecified'):
        """Create a grayscale image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (greyscale values as uint8's).

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        self._encoding = 'mono16'
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the given data is a uint8 array with one channel.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.uint8:
            raise ValueError(
                'Illegal data type. Grayscale images only support 8-bit uint arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError(
                'Illegal data type. Grayscale images only support single channel ')

    def _image_data(self):
        """Returns the data in image format, with scaling and conversion to uint8 types.

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is simply the greyscale entry
            scaled to between 0 and BINARY_IM_MAX_VAL.
        """
        return self._data

    def resize(self, size, interp='bilinear'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')

        Returns
        -------
        :obj:`GrayscaleImage`
            The resized image.
        """
        resized_data = imresize(self.data, size, interp=interp).astype(np.uint8)
        return GrayscaleImage(resized_data, self._frame)

    def to_color(self):
        """Convert the grayscale image to a ColorImage.

        Returns
        -------
        :obj:`ColorImage`
            A color image equivalent to the grayscale one.
        """
        color_data = np.repeat(self.data[:,:,np.newaxis], 3, axis=2)
        return ColorImage(color_data, self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Creates a GrayscaleImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`GrayscaleImage`
            The new grayscale image.
        """
        data = Image.load_data(filename)
        return GrayscaleImage(data, frame)


class BinaryImage(Image):
    """A binary image in which individual pixels are either black or white (0 or BINARY_IM_MAX_VAL).
    """

    def __init__(self, data, frame='unspecified',
                 threshold=BINARY_IM_DEFAULT_THRESH):
        """Create a BinaryImage image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (only one channel, all uint8).
            The data array will be thresholded
            and will end up only containing elements that are BINARY_IM_MAX_VAL or 0.

        threshold : int
            A threshold value. Any value in the data array greater than
            threshold will be set to BINARY_IM_MAX_VAL, and all others will be set to 0.

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        self._encoding = 'passthrough'
        self._threshold = threshold
        data = BINARY_IM_MAX_VAL * \
            (data > threshold).astype(data.dtype)  # binarize
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the given data is a uint8 array with one channel.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.uint8:
            raise ValueError(
                'Illegal data type. Binary images only support 8-bit uint arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError(
                'Illegal data type. Binary images only support single channel ')

    def _image_data(self):
        """Returns the data in image format, with scaling and conversion to uint8 types.

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is simply the binary 0/BINARY_IM_MAX_VAL value.
        """
        return self._data.squeeze()

    def resize(self, size, interp='bilinear'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')

        Returns
        -------
        :obj:`BinaryImage`
            The resized image.
        """
        resized_data = imresize(self.data.astype(np.float32), size, interp=interp)
        return BinaryImage(resized_data.astype(np.uint8), self._frame)

    def mask_binary(self, binary_im):
        """ Takes AND operation with other binary image.

        Parameters
        ----------
        binary_im : :obj:`BinaryImage`
            binary image for and operation

        Returns
        -------
        :obj:`BinaryImage`
            AND of this binary image and other image
        """
        data = np.copy(self._data)
        ind = np.where(binary_im.data == 0)
        data[ind[0], ind[1], ...] = 0
        return BinaryImage(data, self._frame)

    def pixelwise_or(self, binary_im):
        """ Takes OR operation with other binary image.

        Parameters
        ----------
        binary_im : :obj:`BinaryImage`
            binary image for and operation

        Returns
        -------
        :obj:`BinaryImage`
            OR of this binary image and other image
        """
        data = np.copy(self._data)
        ind = np.where(binary_im.data > 0)
        data[ind[0], ind[1], ...] = BINARY_IM_MAX_VAL
        return BinaryImage(data, self._frame)

    def inverse(self):
        """ Inverts image (all nonzeros become zeros and vice verse)
        Returns
        -------
        :obj:`BinaryImage`
            inverse of this binary image
        """
        data = np.zeros(self.shape).astype(np.uint8)
        ind = np.where(self.data == 0)
        data[ind[0], ind[1], ...] = BINARY_IM_MAX_VAL
        return BinaryImage(data, self._frame)

    def prune_contours(self, area_thresh=1000.0, dist_thresh=20,
                       preserve_topology=True):
        """Removes all white connected components with area less than area_thresh.
        Parameters
        ----------
        area_thresh : float
            The minimum area for which a white connected component will not be
            zeroed out.
        dist_thresh : int
            If a connected component is within dist_thresh of the top of the
            image, it will not be pruned out, regardless of its area.
        Returns
        -------
        :obj:`BinaryImage`
            The new pruned binary image.
        """
        # get all contours (connected components) from the binary image
        contours, hierarchy = cv2.findContours(
            self.data.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        num_contours = len(contours)
        middle_pixel = np.array(self.shape)[:2] / 2
        middle_pixel = middle_pixel.reshape(1, 2)
        center_contour = None
        pruned_contours = []

        # find which contours need to be pruned
        for i in range(num_contours):
            area = cv2.contourArea(contours[i])
            if area > area_thresh:
                # check close to origin
                fill = np.zeros([self.height, self.width, 3])
                cv2.fillPoly(
                    fill,
                    pts=[
                        contours[i]],
                    color=(
                        BINARY_IM_MAX_VAL,
                        BINARY_IM_MAX_VAL,
                        BINARY_IM_MAX_VAL))
                nonzero_px = np.where(fill > 0)
                nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
                dists = ssd.cdist(middle_pixel, nonzero_px)
                min_dist = np.min(dists)
                pruned_contours.append((contours[i], min_dist))

        if len(pruned_contours) == 0:
            return None

        pruned_contours.sort(key=lambda x: x[1])

        # keep all contours within some distance of the top
        num_contours = len(pruned_contours)
        keep_indices = [0]
        source_coords = pruned_contours[0][0].squeeze().astype(np.float32)
        for i in range(1, num_contours):
            target_coords = pruned_contours[i][0].squeeze().astype(np.float32)
            dists = ssd.cdist(source_coords, target_coords)
            min_dist = np.min(dists)
            if min_dist < dist_thresh:
                keep_indices.append(i)

        # keep the top num_areas pruned contours
        keep_indices = np.unique(keep_indices)
        pruned_contours = [pruned_contours[i][0] for i in keep_indices]

        # mask out bad areas in the image
        pruned_data = np.zeros([self.height, self.width, 3])
        for contour in pruned_contours:
            cv2.fillPoly(
                pruned_data,
                pts=[contour],
                color=(
                    BINARY_IM_MAX_VAL,
                    BINARY_IM_MAX_VAL,
                    BINARY_IM_MAX_VAL))
        pruned_data = pruned_data[:, :, 0]  # convert back to one channel

        # preserve topology of original image
        if preserve_topology:
            orig_zeros = np.where(self.data == 0)
            pruned_data[orig_zeros[0], orig_zeros[1]] = 0
        return BinaryImage(pruned_data.astype(np.uint8), self._frame)

    def find_contours(self, min_area=0.0, max_area=np.inf):
        """Returns a list of connected components with an area between
        min_area and max_area.
        Parameters
        ----------
        min_area : float
            The minimum area for a contour
        max_area : float
            The maximum area for a contour
        Returns
        -------
        :obj:`list` of :obj:`Contour`
            A list of resuting contours
        """
        # get all contours (connected components) from the binary image
        _, contours, hierarchy = cv2.findContours(
            self.data.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        num_contours = len(contours)
        kept_contours = []

        # find which contours need to be pruned
        for i in range(num_contours):
            area = cv2.contourArea(contours[i])
            logging.debug('Contour %d area: %.3f' % (len(kept_contours), area))
            if area > min_area and area < max_area:
                boundary_px = contours[i].squeeze()
                boundary_px_ij_swapped = np.zeros(boundary_px.shape)
                boundary_px_ij_swapped[:, 0] = boundary_px[:, 1]
                boundary_px_ij_swapped[:, 1] = boundary_px[:, 0]
                kept_contours.append(
                    Contour(
                        boundary_px_ij_swapped,
                        area=area,
                        frame=self._frame))

        return kept_contours

    def contour_mask(self, contour):
        """ Generates a binary image with only the given contour filled in. """
        # fill in new data
        new_data = np.zeros(self.data.shape)
        num_boundary = contour.boundary_pixels.shape[0]
        boundary_px_ij_swapped = np.zeros([num_boundary, 1, 2])
        boundary_px_ij_swapped[:, 0, 0] = contour.boundary_pixels[:, 1]
        boundary_px_ij_swapped[:, 0, 1] = contour.boundary_pixels[:, 0]
        cv2.fillPoly(
            new_data, pts=[
                boundary_px_ij_swapped.astype(
                    np.int32)], color=(
                BINARY_IM_MAX_VAL, BINARY_IM_MAX_VAL, BINARY_IM_MAX_VAL))
        orig_zeros = np.where(self.data == 0)
        new_data[orig_zeros[0], orig_zeros[1]] = 0
        return BinaryImage(new_data.astype(np.uint8), frame=self._frame)

    def boundary_map(self):
        """ Computes the boundary pixels in the image and sets them to nonzero values.

        Returns
        -------
        :obj:`BinaryImage`
            binary image with nonzeros on the boundary of the original image
        """
        # compute contours
        contours = self.find_contours()

        # fill in nonzero pixels
        new_data = np.zeros(self.data.shape)
        for contour in contours:
            new_data[contour.boundary_pixels[:, 0].astype(np.uint8),
                     contour.boundary_pixels[:, 1].astype(np.uint8)] = np.iinfo(np.uint8).max
        return BinaryImage(new_data.astype(np.uint8), frame=self.frame)

    def closest_pixel_to_set(self, start, pixel_set, direction, w=13, t=0.5):
        """Starting at pixel, moves start by direction * t until there is a
        pixel from pixel_set within a radius w of start. Then, returns start.

        Parameters
        ----------
        start : :obj:`numpy.ndarray` of float
            The initial pixel location at which to start.

        pixel_set : set of 2-tuples of float
            The set of pixels to check set intersection with

        direction : :obj:`numpy.ndarray` of float
            The 2D direction vector in which to move pixel.

        w : int
            A circular diameter in which to check for pixels.
            As soon as the current pixel has some non-zero pixel with a diameter
            w of it, this function returns the current pixel location.

        t : float
            The step size with which to move pixel along direction.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The first pixel location along the direction vector at which there
            exists some intersection with pixel_set within a radius w.
        """

        # create circular structure for checking clearance
        y, x = np.meshgrid(np.arange(w) - w / 2, np.arange(w) - w / 2)
        cur_px_y = np.ravel(y + start[0]).astype(np.uint16)
        cur_px_x = np.ravel(x + start[1]).astype(np.uint16)
        
        # create comparison set, check set overlap
        cur_px = set(zip(cur_px_y, cur_px_x))
        includes = True
        if np.all(
            cur_px_y >= 0) and np.all(
            cur_px_y < self.height) and np.all(
            cur_px_x >= 0) and np.all(
                cur_px_x < self.width):
            includes = not cur_px.isdisjoint(pixel_set)
        else:
            return None

        # Continue until out of bounds or sets overlap
        while not includes:
            start = start + t * direction
            cur_px_y = np.ravel(y + start[0]).astype(np.uint16)
            cur_px_x = np.ravel(x + start[1]).astype(np.uint16)
            cur_px = set(zip(cur_px_y, cur_px_x))
            if np.all(
                cur_px_y >= 0) and np.all(
                cur_px_y < self.height) and np.all(
                cur_px_x >= 0) and np.all(
                    cur_px_x < self.width):
                includes = not cur_px.isdisjoint(pixel_set)
            else:
                return None
        
        return start

    def closest_nonzero_pixel(self, pixel, direction, w=13, t=0.5):
        """Starting at pixel, moves pixel by direction * t until there is a
        non-zero pixel within a radius w of pixel. Then, returns pixel.

        Parameters
        ----------
        pixel : :obj:`numpy.ndarray` of float
            The initial pixel location at which to start.

        direction : :obj:`numpy.ndarray` of float
            The 2D direction vector in which to move pixel.

        w : int
            A circular diameter in which to check for non-zero pixels.
            As soon as the current pixel has some non-zero pixel with a diameter
            w of it, this function returns the current pixel location.

        t : float
            The step size with which to move pixel along direction.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The first pixel location along the direction vector at which there
            exists some non-zero pixel within a radius w.
        """
        # create circular structure for checking clearance
        y, x = np.meshgrid(np.arange(w) - w / 2, np.arange(w) - w / 2)

        cur_px_y = np.ravel(y + pixel[0]).astype(np.uint16)
        cur_px_x = np.ravel(x + pixel[1]).astype(np.uint16)
        occupied = False
        if np.all(
            cur_px_y >= 0) and np.all(
            cur_px_y < self.height) and np.all(
            cur_px_x >= 0) and np.all(
                cur_px_x < self.width):
            occupied = np.any(self[cur_px_y, cur_px_x] >= self._threshold)
        else:
            return None 

        while not occupied:
            pixel = pixel + t * direction
            cur_px_y = np.ravel(y + pixel[0]).astype(np.uint16)
            cur_px_x = np.ravel(x + pixel[1]).astype(np.uint16)
            if np.all(
                cur_px_y >= 0) and np.all(
                cur_px_y < self.height) and np.all(
                cur_px_x >= 0) and np.all(
                    cur_px_x < self.width):
                occupied = np.any(self[cur_px_y, cur_px_x] >= self._threshold)
            else:
                return None

        return pixel
    
    def closest_allzero_pixel(self, pixel, direction, w=13, t=0.5):
        """Starting at pixel, moves pixel by direction * t until all
        zero pixels within a radius w of pixel. Then, returns pixel.

        Parameters
        ----------
        pixel : :obj:`numpy.ndarray` of float
            The initial pixel location at which to start.

        direction : :obj:`numpy.ndarray` of float
            The 2D direction vector in which to move pixel.

        w : int
            A circular diameter in which to check for zero pixels.
            As soon as the current pixel has all zero pixels with a diameter
            w of it, this function returns the current pixel location.

        t : float
            The step size with which to move pixel along direction.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The first pixel location along the direction vector at which there
            exists all zero pixels within a radius w.
        """
        # create circular structure for checking clearance
        y, x = np.meshgrid(np.arange(w) - w / 2, np.arange(w) - w / 2)

        cur_px_y = np.ravel(y + pixel[0]).astype(np.uint16)
        cur_px_x = np.ravel(x + pixel[1]).astype(np.uint16)
        
        # Check if all pixels in radius are in bounds and zero-valued
        empty = False
        if np.all(
            cur_px_y >= 0) and np.all(
            cur_px_y < self.height) and np.all(
            cur_px_x >= 0) and np.all(
                cur_px_x < self.width):
            empty = np.all(self[cur_px_y, cur_px_x] <= self._threshold)
        else:
            return None
        
        # If some nonzero pixels, continue incrementing along direction
        # and checking for empty space
        while not empty:
            pixel = pixel + t * direction
            cur_px_y = np.ravel(y + pixel[0]).astype(np.uint16)
            cur_px_x = np.ravel(x + pixel[1]).astype(np.uint16)
            if np.all(
                cur_px_y >= 0) and np.all(
                cur_px_y < self.height) and np.all(
                cur_px_x >= 0) and np.all(
                    cur_px_x < self.width):
                empty = np.all(self[cur_px_y, cur_px_x] <= self._threshold)
            else:
                return None
                
        return pixel

    def add_frame(
            self,
            left_boundary,
            right_boundary,
            upper_boundary,
            lower_boundary):
        """ Adds a frame to the image, e.g. turns the boundaries white

        Parameters
        ----------
        left_boundary : int
            the leftmost boundary of the frame
        right_boundary : int
            the rightmost boundary of the frame (must be greater than left_boundary)
        upper_boundary : int
            the upper boundary of the frame
        lower_boundary : int
            the lower boundary of the frame (must be greater than upper_boundary)

        Returns
        -------
        :obj:`BinaryImage`
            binary image with white (BINARY_IM_MAX_VAL) on the boundaries
        """
        # check valid boundary pixels
        left_boundary = max(0, left_boundary)
        right_boundary = min(self.width - 1, right_boundary)
        upper_boundary = max(0, upper_boundary)
        lower_boundary = min(self.height - 1, lower_boundary)

        if right_boundary < left_boundary:
            raise ValueError(
                'Left boundary must be smaller than the right boundary')
        if upper_boundary > lower_boundary:
            raise ValueError(
                'Upper boundary must be smaller than the lower boundary')

        # fill in border pixels
        bordered_data = self.data.copy()
        bordered_data[:upper_boundary, :] = BINARY_IM_MAX_VAL
        bordered_data[lower_boundary:, :] = BINARY_IM_MAX_VAL
        bordered_data[:, :left_boundary] = BINARY_IM_MAX_VAL
        bordered_data[:, right_boundary:] = BINARY_IM_MAX_VAL
        return BinaryImage(bordered_data, frame=self._frame)

    def to_distance_im(self):
        """ Returns the distance-transformed image as a raw float array.

        Returns
        -------
        :obj:`numpy.ndarray`
            HxW float array containing the distance transform of the binary image
        """
        return snm.distance_transform_edt(BINARY_IM_MAX_VAL - self.data)
        
    def most_free_pixel(self):
        """ Find the black pixel with the largest distance from the white pixels.

        Returns
        -------
        :obj:`numpy.ndarray`
            2-vector containing the most free pixel
        """
        dist_tf = self.to_distance_im()
        max_px = np.where(dist_tf == np.max(dist_tf))
        free_pixel = np.array([max_px[0][0], max_px[1][0]])
        return free_pixel
    
    def diff_with_target(self, binary_im):
        """ Creates a color image to visualize the overlap between two images.
        Nonzero pixels that match in both images are green.
        Nonzero pixels of this image that aren't in the other image are yellow
        Nonzero pixels of the other image that aren't in this image are red

        Parameters
        ----------
        binary_im : :obj:`BinaryImage`
            binary image to take the difference with

        Returns
        -------
        :obj:`ColorImage`
            color image to visualize the image difference
        """
        red = np.array([BINARY_IM_MAX_VAL, 0, 0])
        yellow = np.array([BINARY_IM_MAX_VAL, BINARY_IM_MAX_VAL, 0])
        green = np.array([0, BINARY_IM_MAX_VAL, 0])
        overlap_data = np.zeros([self.height, self.width, 3])
        unfilled_px = np.where((self.data == 0) & (binary_im.data > 0))
        overlap_data[unfilled_px[0], unfilled_px[1], :] = red
        filled_px = np.where((self.data > 0) & (binary_im.data > 0))
        overlap_data[filled_px[0], filled_px[1], :] = green
        spurious_px = np.where((self.data > 0) & (binary_im.data == 0))
        overlap_data[spurious_px[0], spurious_px[1], :] = yellow
        return ColorImage(overlap_data.astype(np.uint8), frame=self.frame)

    def num_adjacent(self, i, j):
        """ Counts the number of adjacent nonzero pixels to a given pixel.

        Parameters
        ----------
        i : int
            row index of query pixel
        j : int
            col index of query pixel

        Returns
        -------
        int
            number of adjacent nonzero pixels
        """
        # check values
        if i < 1 or i > self.height - 2 or j < 1 and j > self.width - 2:
            raise ValueError('Pixels out of bounds')

        # count the number of blacks
        count = 0
        diffs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for d in diffs:
            if self.data[i + d[0]][j + d[1]] > self._threshold:
                count += 1
        return count

    def to_sdf(self):
        """ Converts the 2D image to a 2D signed distance field.

        Returns
        -------
        :obj:`numpy.ndarray`
            2D float array of the signed distance field
        """
        # compute medial axis transform
        skel, sdf_in = morph.medial_axis(self.data, return_distance=True)
        useless_skel, sdf_out = morph.medial_axis(
            np.iinfo(np.uint8).max - self.data, return_distance=True)

        # convert to true sdf
        sdf = sdf_out - sdf_in
        return sdf

    def to_color(self):
        """Creates a ColorImage from the binary image.

        Returns
        -------
        :obj:`ColorImage`
            The newly-created color image.
        """
        color_data = np.zeros([self.height, self.width, 3])
        color_data[:, :, 0] = self.data
        color_data[:, :, 1] = self.data
        color_data[:, :, 2] = self.data
        return ColorImage(color_data.astype(np.uint8), self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Creates a BinaryImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`BinaryImage`
            The new binary image.
        """
        data = Image.load_data(filename)
        if len(data.shape) > 2 and data.shape[2] > 1:
            data = data[:, :, 0]
        return BinaryImage(data, frame)


class RgbdImage(Image):
    """ An image containing a red, green, blue, and depth channel. """

    def __init__(self, data, frame='unspecified'):
        """ Create an RGB-D image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (four channels, all float).
            The first three channels should be the red, greed, and blue channels
            which must be in the range (0, BINARY_IM_MAX_VAL).
            The fourth channel should be the depth channel.

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the given data is a float array with four channels.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.float32 and \
           data.dtype.type is not np.float64:
            raise ValueError(
                'Illegal data type. RGB-D images only support float arrays')

        if len(data.shape) != 3 and data.shape[2] != 4:
            raise ValueError(
                'Illegal data type. RGB-D images only support four channel')

        color_data = data[:, :, :3]
        if np.any((color_data < 0) | (color_data > BINARY_IM_MAX_VAL)):
            raise ValueError(
                'Color channels must be in the range (0, BINARY_IM_MAX_VAL)')

    @staticmethod
    def from_color_and_depth(color_im, depth_im):
        """ Creates an RGB-D image from a separate color and depth image. """
        # check shape
        if color_im.height != depth_im.height or color_im.width != depth_im.width:
            raise ValueError('Color and depth images must have the same shape')

        # check frame
        if color_im.frame != depth_im.frame:
            raise ValueError('Color and depth images must have the same frame')

        # form composite data
        rgbd_data = np.zeros([color_im.height, color_im.width, 4])
        rgbd_data[:, :, :3] = color_im.data.astype(np.float64)
        rgbd_data[:, :, 3] = depth_im.data
        return RgbdImage(rgbd_data, frame=color_im.frame)

    @property
    def color(self):
        """ Returns the color image. """
        return ColorImage(self.raw_data[:, :, :3].astype(
            np.uint8), frame=self.frame)

    @property
    def depth(self):
        """ Returns the depth image. """
        return DepthImage(self.raw_data[:, :, 3], frame=self.frame)

    def _image_data(self, normalize=False):
        """Returns the data in image format, with scaling and conversion to uint8 types.
        NOTE: Only returns the color image!!!!

        Parameters
        ----------
        normalize : bool
            whether or not to normalize by the min and max depth of the image

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is a set of 3 RGB values, each of
            which is simply the depth entry scaled to between 0 and BINARY_IM_MAX_VAL.
        """
        return self.color_im._image_data(normalize=normalize)

    def mask_binary(self, binary_im):
        """Create a new image by zeroing out data at locations
        where binary_im == 0.0.

        Parameters
        ----------
        binary_im : :obj:`BinaryImage`
            A BinaryImage of the same size as this image, with pixel values of either
            zero or one. Wherever this image has zero pixels, we'll zero out the
            pixels of the new image.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type, masked by the given binary image.
        """
        color = self.color.mask_binary(binary_im)
        depth = self.depth.mask_binary(binary_im)
        return RgbdImage.from_color_and_depth(color, depth)
    
    def resize(self, size, interp='bilinear'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')
        """
        # resize channels separately
        color_im_resized = self.color.resize(size, interp)
        depth_im_resized = self.depth.resize(size, interp)

        # return combination of resized data
        return RgbdImage.from_color_and_depth(
            color_im_resized, depth_im_resized)

    def crop(self, height, width, center_i=None, center_j=None):
        """Crop the image centered around center_i, center_j.

        Parameters
        ----------
        height : int
            The height of the desired image.

        width : int
            The width of the desired image.

        center_i : int
            The center height point at which to crop. If not specified, the center
            of the image is used.

        center_j : int
            The center width point at which to crop. If not specified, the center
            of the image is used.

        Returns
        -------
        :obj:`Image`
            A cropped Image of the same type.
        """
        # crop channels separately
        color_im_cropped = self.color.crop(height, width,
                                           center_i=center_i,
                                           center_j=center_j)
        depth_im_cropped = self.depth.crop(height, width,
                                           center_i=center_i,
                                           center_j=center_j)

        # return combination of cropped data
        return RgbdImage.from_color_and_depth(
            color_im_cropped, depth_im_cropped)

    def transform(self, translation, theta, method='opencv'):
        """Create a new image by translating and rotating the current image.

        Parameters
        ----------
        translation : :obj:`numpy.ndarray` of float
            The XY translation vector.
        theta : float
            Rotation angle in radians, with positive meaning counter-clockwise.
        method : :obj:`str`
            Method to use for image transformations (opencv or scipy)

        Returns
        -------
        :obj:`Image`
            An image of the same type that has been rotated and translated.
        """
        # transform channels separately
        color_im_tf = self.color.transform(translation, theta, method=method)
        depth_im_tf = self.depth.transform(translation, theta, method=method)

        # return combination of cropped data
        return RgbdImage.from_color_and_depth(color_im_tf, depth_im_tf)

    def to_grayscale_depth(self):
        """ Converts to a grayscale and depth (G-D) image. """
        gray = self.color.to_grayscale()
        return GdImage.from_grayscale_and_depth(gray, self.depth)

    def combine_with(self, rgbd_im):
        """
        Replaces all zeros in the source rgbd image with the values of a different rgbd image

        Parameters
        ----------
        rgbd_im : :obj:`RgbdImage`
            rgbd image to combine with

        Returns
        -------
        :obj:`RgbdImage`
            the combined rgbd image
        """
        new_data = self.data.copy()
        depth_data = self.depth.data
        other_depth_data = rgbd_im.depth.data
        depth_zero_px = self.depth.zero_pixels()
        depth_replace_px = np.where(
            (other_depth_data != 0) & (
                other_depth_data < depth_data))
        depth_replace_px = np.c_[depth_replace_px[0], depth_replace_px[1]]

        # replace zero pixels
        new_data[depth_zero_px[:, 0], depth_zero_px[:, 1],
                 :] = rgbd_im.data[depth_zero_px[:, 0], depth_zero_px[:, 1], :]

        # take closest pixel
        new_data[depth_replace_px[:, 0], depth_replace_px[:, 1],
                 :] = rgbd_im.data[depth_replace_px[:, 0], depth_replace_px[:, 1], :]

        return RgbdImage(new_data, frame=self.frame)

    def crop(self, height, width, center_i=None, center_j=None):
        """Crop the image centered around center_i, center_j.

        Parameters
        ----------
        height : int
            The height of the desired image.

        width : int
            The width of the desired image.

        center_i : int
            The center height point at which to crop. If not specified, the center
            of the image is used.

        center_j : int
            The center width point at which to crop. If not specified, the center
            of the image is used.

        Returns
        -------
        :obj:`Image`
            A cropped Image of the same type.
        """
        color_im_crop = self.color.crop(height, width, center_i, center_j)
        depth_im_crop = self.depth.crop(height, width, center_i, center_j)
        return RgbdImage.from_color_and_depth(color_im_crop, depth_im_crop)


class GdImage(Image):
    """ An image containing a grayscale and depth channel. """

    def __init__(self, data, frame='unspecified'):
        """Create a G-D image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (two channels, both float).
            The first channel should be the grayscale channel
            which must be in the range (0, BINARY_IM_MAX_VAL).
            The second channel should be the depth channel.

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the given data is a float array with four channels.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.float32 and \
           data.dtype.type is not np.float64:
            raise ValueError(
                'Illegal data type. G-D images only support float arrays')

        if len(data.shape) != 3 and data.shape[2] != 2:
            raise ValueError(
                'Illegal data type. G-D images only support two channel')

        gray_data = data[:, :, 0]
        if np.any((gray_data < 0) | (gray_data > BINARY_IM_MAX_VAL)):
            raise ValueError(
                'Gray channel must be in the range (0, BINARY_IM_MAX_VAL)')

    @staticmethod
    def from_grayscale_and_depth(gray_im, depth_im):
        """ Creates an G-D image from a separate grayscale and depth image. """
        # check shape
        if gray_im.height != depth_im.height or gray_im.width != depth_im.width:
            raise ValueError(
                'Grayscale and depth images must have the same shape')

        # check frame
        if gray_im.frame != depth_im.frame:
            raise ValueError(
                'Grayscale and depth images must have the same frame')

        # form composite data
        gd_data = np.zeros([gray_im.height, gray_im.width, 2])
        gd_data[:, :, 0] = gray_im.data.astype(np.float64)
        gd_data[:, :, 1] = depth_im.data
        return GdImage(gd_data, frame=gray_im.frame)

    @property
    def gray(self):
        """ Returns the grayscale image. """
        return GrayscaleImage(
            self.raw_data[:, :, 0].astype(np.uint8), frame=self.frame)

    @property
    def depth(self):
        """ Returns the depth image. """
        return DepthImage(self.raw_data[:, :, 1], frame=self.frame)

    def _image_data(self, normalize=False):
        """Returns the data in image format, with scaling and conversion to uint8 types.
        NOTE: Only returns the color image!!!!

        Parameters
        ----------
        normalize : bool
            whether or not to normalize by the min and max depth of the image

        Returns
        -------
        :obj:`numpy.ndarray` of uint8
            A 3D matrix representing the image. The first dimension is rows, the
            second is columns, and the third is a set of 3 RGB values, each of
            which is simply the depth entry scaled to between 0 and BINARY_IM_MAX_VAL.
        """
        return self.gray_im._image_data(normalize=normalize)

    def resize(self, size, interp='bilinear'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')
        """
        # resize channels separately
        gray_im_resized = self.gray.resize(size, interp)
        depth_im_resized = self.depth.resize(size, interp)

        # return combination of resized data
        return GdImage.from_grayscale_and_depth(
            gray_im_resized, depth_im_resized)

    def crop(self, height, width, center_i=None, center_j=None):
        """Crop the image centered around center_i, center_j.

        Parameters
        ----------
        height : int
            The height of the desired image.

        width : int
            The width of the desired image.

        center_i : int
            The center height point at which to crop. If not specified, the center
            of the image is used.

        center_j : int
            The center width point at which to crop. If not specified, the center
            of the image is used.

        Returns
        -------
        :obj:`Image`
            A cropped Image of the same type.
        """
        gray_im_crop = self.gray.crop(height, width, center_i, center_j)
        depth_im_crop = self.depth.crop(height, width, center_i, center_j)
        return GdImage.from_grayscale_and_depth(gray_im_crop, depth_im_crop)


class SegmentationImage(Image):
    """An image containing integer-valued segment labels.
    """

    def __init__(self, data, frame='unspecified'):
        """Create a Segmentation image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (only one channel, all uint8).
            The integer-valued data should correspond to segment labels.

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        self._num_segments = np.max(data) + 1
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """ Checks for uint8, single channel """
        if data.dtype.type is not np.uint8 and data.dtype.type is not np.uint16:
            raise ValueError(
                'Illegal data type. Segmentation images only support 8-bit or 16-bit uint arrays')

        if len(data.shape) == 3 and data.shape[2] != 1:
            raise ValueError(
                'Illegal data type. Segmentation images only support single channel ')

    @property
    def num_segments(self):
        return self._num_segments

    def _image_data(self):
        return self._data

    def border_pixels(
            self,
            grad_sigma=0.5,
            grad_lower_thresh=0.1,
            grad_upper_thresh=1.0):
        """
        Returns the pixels on the boundary between all segments, excluding the zero segment.

        Parameters
        ----------
        grad_sigma : float
            standard deviation used for gaussian gradient filter
        grad_lower_thresh : float
            lower threshold on gradient threshold used to determine the boundary pixels
        grad_upper_thresh : float
            upper threshold on gradient threshold used to determine the boundary pixels

        Returns
        -------
        :obj:`numpy.ndarray`
             Nx2 array of pixels on the boundary
        """
        # boundary pixels
        boundary_im = np.ones(self.shape)
        for i in range(1, self.num_segments):
            label_border_im = self.data.copy()
            label_border_im[self.data == 0] = i
            grad_mag = sf.gaussian_gradient_magnitude(
                label_border_im.astype(np.float32), sigma=grad_sigma)

            nonborder_px = np.where(
                (grad_mag < grad_lower_thresh) | (
                    grad_mag > grad_upper_thresh))
            boundary_im[nonborder_px[0], nonborder_px[1]] = 0

        # return boundary pixels
        border_px = np.where(boundary_im > 0)
        border_px = np.c_[border_px[0], border_px[1]]
        return border_px

    def segment_mask(self, segnum):
        """ Returns a binary image of just the segment corresponding to the given number.

        Parameters
        ----------
        segnum : int
            the number of the segment to generate a mask for

        Returns
        -------
        :obj:`BinaryImage`
             binary image data
        """
        binary_data = np.zeros(self.shape)
        binary_data[self.data == segnum] = BINARY_IM_MAX_VAL
        return BinaryImage(binary_data.astype(np.uint8), frame=self.frame)

    def mask_binary(self, binary_im):
        """Create a new image by zeroing out data at locations
        where binary_im == 0.0.

        Parameters
        ----------
        binary_im : :obj:`BinaryImage`
            A BinaryImage of the same size as this image, with pixel values of either
            zero or one. Wherever this image has zero pixels, we'll zero out the
            pixels of the new image.

        Returns
        -------
        :obj:`Image`
            A new Image of the same type, masked by the given binary image.
        """
        data = np.copy(self._data)
        ind = np.where(binary_im.data == 0)
        data[ind[0], ind[1], :] = 0
        return SegmentationImage(data, self._frame)
    
    def resize(self, size, interp='nearest'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')
        """
        resized_data = imresize(self.data, size, interp=interp).astype(np.uint8)
        return SegmentationImage(resized_data, self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """ Opens a segmentation image """
        data = Image.load_data(filename)
        return SegmentationImage(data, frame)


class PointCloudImage(Image):
    """A point cloud image in which individual pixels have three float channels.
    """

    def __init__(self, data, frame='unspecified'):
        """Create a PointCloudImage image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (three floats).

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the given data is a float array with three channels.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.float32 and data.dtype.type is not np.float64:
            raise ValueError(
                'Illegal data type. PointCloud images only support 32-bit or 64-bit float arrays')

        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError(
                'Illegal data type. PointCloud images must have three channels')

    def _image_data(self):
        """This method is not implemented for PointCloudImages.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            'Image conversion not supported for point cloud')

    def resize(self, size, interp='nearest'):
        """Resize the image.

        Parameters
        ----------
        size : int, float, or tuple
            * int   - Percentage of current size.
            * float - Fraction of current size.
            * tuple - Size of the output image.

        interp : :obj:`str`, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic', or 'cubic')

        Returns
        -------
        :obj:`PointCloudImage`
            The resized image.
        """
        resized_data_0 = imresize(self._data[:,:,0], size, interp=interp).astype(np.float32)
        resized_data_1 = imresize(self._data[:,:,1], size, interp=interp).astype(np.float32)
        resized_data_2 = imresize(self._data[:,:,2], size, interp=interp).astype(np.float32)
        resized_data = np.zeros([resized_data_0.shape[0],
                                 resized_data_0.shape[1],
                                 self.channels])
        resized_data[:,:,0] = resized_data_0
        resized_data[:,:,1] = resized_data_1
        resized_data[:,:,2] = resized_data_2
        return PointCloudImage(resized_data, self._frame)

    def to_mesh(self, dist_thresh=0.01):
        """ Convert the point cloud to a mesh.

        Returns
        -------
        :obj:`trimesh.Trimesh`
            mesh of the point cloud
        """
        # init vertex and triangle buffers
        vertices = []
        triangles = []
        vertex_indices = -1 * np.ones([self.height, self.width]).astype(np.int32)
        
        for i in range(self.height-1):
            for j in range(self.width-1):
                # read corners of square
                v0 = self.data[i,j,:]
                v1 = self.data[i,j+1,:]
                v2 = self.data[i+1,j,:]
                v3 = self.data[i+1,j+1,:]

                # check distances
                d01 = np.abs(v0[2] - v1[2])
                d02 = np.abs(v0[2] - v2[2])
                d03 = np.abs(v0[2] - v3[2])
                d13 = np.abs(v1[2] - v3[2])
                d23 = np.abs(v2[2] - v3[2])

                # add tri 1
                if max(d01, d03, d13) < dist_thresh:
                    # add vertices
                    if vertex_indices[i,j] == -1:
                        vertices.append(v0)
                        vertex_indices[i,j] = len(vertices)-1
                    if vertex_indices[i,j+1] == -1:
                        vertices.append(v1)
                        vertex_indices[i,j+1] = len(vertices)-1
                    if vertex_indices[i+1,j+1] == -1:
                        vertices.append(v3)
                        vertex_indices[i+1,j+1] = len(vertices)-1
                
                    # add tri
                    i0 = vertex_indices[i,j]
                    i1 = vertex_indices[i,j+1]
                    i3 = vertex_indices[i+1,j+1]
                    triangles.append([i0, i1, i3])

                # add tri 2
                if max(d01, d03, d23) < dist_thresh:
                    # add vertices
                    if vertex_indices[i,j] == -1:
                        vertices.append(v0)
                        vertex_indices[i,j] = len(vertices)-1
                    if vertex_indices[i+1,j] == -1:
                        vertices.append(v2)
                        vertex_indices[i+1,j] = len(vertices)-1
                    if vertex_indices[i+1,j+1] == -1:
                        vertices.append(v3)
                        vertex_indices[i+1,j+1] = len(vertices)-1
                
                    # add tri
                    i0 = vertex_indices[i,j]
                    i2 = vertex_indices[i+1,j]
                    i3 = vertex_indices[i+1,j+1]
                    triangles.append([i0, i3, i2])

        # return trimesh
        import trimesh
        mesh = trimesh.Trimesh(vertices, triangles)
        return mesh
        
    def to_point_cloud(self):
        """Convert the image to a PointCloud object.

        Returns
        -------
        :obj:`autolab_core.PointCloud`
            The corresponding PointCloud.
        """
        return PointCloud(
            data=self._data.reshape(
                self.height *
                self.width,
                3).T,
            frame=self._frame)
    
    def normal_cloud_im(self, ksize=3):
        """Generate a NormalCloudImage from the PointCloudImage using Sobel filtering.

        Parameters
        ----------
        ksize : int
            Size of the kernel to use for derivative computation

        Returns
        -------
        :obj:`NormalCloudImage`
            The corresponding NormalCloudImage.
        """
        # compute direction via cross product of derivatives
        gy = cv2.Sobel(self.data, cv2.CV_64F, 1, 0, ksize=ksize)
        gx = cv2.Sobel(self.data, cv2.CV_64F, 0, 1, ksize=ksize)
        gx_data = gx.reshape(self.height * self.width, 3)
        gy_data = gy.reshape(self.height * self.width, 3)
        pc_grads = np.cross(gx_data, gy_data)  # default to point toward camera

        # normalize
        pc_grad_norms = np.linalg.norm(pc_grads, axis=1)
        pc_grads[pc_grad_norms > 0] = pc_grads[pc_grad_norms > 0] / np.tile(pc_grad_norms[pc_grad_norms > 0, np.newaxis], [1, 3])
        pc_grads[pc_grad_norms == 0.0] = np.array([0,0,-1.0]) # zero norm means pointing toward camera

        # reshape
        normal_im_data = pc_grads.reshape(self.height, self.width, 3)

        # preserve zeros
        zero_px = self.zero_pixels()
        normal_im_data[zero_px[:,0], zero_px[:,1], :] = np.zeros(3)
        
        return NormalCloudImage(normal_im_data, frame=self.frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Creates a PointCloudImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`PointCloudImage`
            The new PointCloudImage.
        """
        data = Image.load_data(filename)
        return PointCloudImage(data, frame)


class NormalCloudImage(Image):
    """A normal cloud image in which individual pixels have three float channels.
    """

    def __init__(self, data, frame='unspecified'):
        """Create a NormalCloudImage image from an array of data.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            An array of data with which to make the image. The first dimension
            of the data should index rows, the second columns, and the third
            individual pixel elements (three floats).

        frame : :obj:`str`
            A string representing the frame of reference in which this image
            lies.

        Raises
        ------
        ValueError
            If the data is not a properly-formatted ndarray or frame is not a
            string.
        """
        Image.__init__(self, data, frame)

    def _check_valid_data(self, data):
        """Checks that the given data is a float array with three channels.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            The data to check.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        if data.dtype.type is not np.float32 and data.dtype.type is not np.float64:
            raise ValueError(
                'Illegal data type. NormalCloud images only support 32-bit or 64-bit float arrays')

        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError(
                'Illegal data type. NormalCloud images must have three channels')

        if np.any((np.abs(np.linalg.norm(data, axis=2) - 1.0) > 1e-4)
                  & (np.linalg.norm(data, axis=2) != 0.0)):
            raise ValueError('Illegal data. Must have norm=1.0 or norm=0.0')

    def _image_data(self):
        """This method is not implemented for NormalCloudImage.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            'Image conversion not supported for normal cloud')

    def resize(self, size, interp='bilinear'):
        """This method is not implemented for NormalCloudImage.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            'Image resizing not supported for normal cloud')

    def to_normal_cloud(self):
        """Convert the image to a NormalCloud object.

        Returns
        -------
        :obj:`autolab_core.NormalCloud`
            The corresponding NormalCloud.
        """
        return NormalCloud(
            data=self._data.reshape(
                self.height *
                self.width,
                3).T,
            frame=self._frame)

    @staticmethod
    def open(filename, frame='unspecified'):
        """Creates a NormalCloudImage from a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the data from. Must be one of .png, .jpg,
            .npy, or .npz.

        frame : :obj:`str`
            A string representing the frame of reference in which the new image
            lies.

        Returns
        -------
        :obj:`NormalCloudImage`
            The new NormalCloudImage.
        """
        data = Image.load_data(filename)
        return NormalCloudImage(data, frame)
