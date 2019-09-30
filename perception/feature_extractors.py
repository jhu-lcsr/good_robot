"""
Classes for image feature extraction.
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import logging
import numpy as np
import time

from .constants import *
from .cnn import AlexNet
from .image import Image, ColorImage, DepthImage

class FeatureExtractor:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def extract(self, image):
        """
        Returns a set of extracted features for an image
        Parameters
        ----------
        image : :obj:`Image`
            object to extract for
        """
        pass

class CNNBatchFeatureExtractor(FeatureExtractor):
    """ Extract feature descriptors for images in a giant batch using Convolutional Neural Networks.

    Attributes
    ----------
    cnn : :obj:`AlexNet`
        the convolutional neural network to use
    """
    def __init__(self, config):
        self.cnn_ = AlexNet(config, use_default_weights=True)

    def open(self):
        """ Opens the tensorflow session. For memory management. """
        self.cnn_.open_session()

    def close(self):
        """ Closes the tensorflow session. For memory management. """
        self.cnn_.close_session()

    def _forward_pass(self, images):
        """ Forward pass a list of images through the CNN """
        # form image array
        num_images = len(images)
        if num_images == 0:
            return None
        for image in images:
            if not isinstance(image, Image):
                new_images = []
                for image in images:
                    if len(image.shape) > 2:
                        new_images.append(ColorImage(image, frame='unspecified'))
                    elif image.dtype == np.float32 or image.dtype == np.float64:
                        new_images.append(DepthImage(image, frame='unspecified'))
                    else:
                        raise ValueError('Image type not understood')
                images = new_images
                break

        im_height = images[0].height
        im_width = images[0].width
        channels = images[0].channels
        tensor_channels = 3
        image_arr = np.zeros([num_images, im_height, im_width, tensor_channels])
        for j, image in enumerate(images):
            if channels == 3:
                image_arr[j,:,:,:] = image.raw_data
            else:
                image_arr[j,:,:,:] = np.tile(image.raw_data, [1,1,1,3])

        # predict
        fp_start = time.time()
        final_blobs = self.cnn_.featurize(image_arr)
        fp_stop = time.time()
        logging.debug('Featurization took %f sec per image' %((fp_stop - fp_start) / len(images)))
        return final_blobs.reshape(final_blobs.shape[0], -1)

    def extract(self, images):
        """ Form feature descriptors for a set of images.

        Parameters
        ----------
        images : :obj:`list` of :obj`Image` or :obj`numpy.ndarray`
            images to extract features for
        """
        return self._forward_pass(images)

class CNNReusableBatchFeatureExtractor(CNNBatchFeatureExtractor):
    """ Extract feature descriptors for images in a giant batch. Allows you to initialize the extractor with a pre-existing CNN, for memory management reasons. """
    def __init__(self, cnn):
        self.cnn_ = cnn
