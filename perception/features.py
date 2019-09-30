"""
Classes for features of a 3D object surface.
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import numpy as np

class Feature:
    """ Abstract class for features """
    __metaclass__ = ABCMeta
    def __init__(self):
        pass

class LocalFeature(Feature):
    """ Local (e.g. pointwise) features on shape surfaces.

    Attributes
    ----------
    descriptor : :obj:`numpy.ndarray`
        vector to describe the point
    reference_frame : :obj:`numpy.ndarray`
        reference frame of the descriptor, as an array
    point : :obj:`numpy.ndarray`
        3D point on shape surface that descriptor corresponds to
    normal : :obj:`numpy.ndarray`
        3D surface normal on shape surface at corresponding point
    """
    __metaclass__ = ABCMeta

    def __init__(self, descriptor, rf, point, normal):
        self.descriptor_ = descriptor
        self.rf_ = rf
        self.point_ = point
        self.normal_ = normal

    @property
    def descriptor(self):
        return self.descriptor_

    @property
    def reference_frame(self):
        return self.rf_

    @property
    def keypoint(self):
        return self.point_

    @property
    def normal(self):
        return self.normal_

class GlobalFeature(Feature):
    """ Global features of a full shape surface.

    Attributes
    ----------
    key : :obj:`str`
        object key in database that descriptor corresponds to
    descriptor : :obj:`numpy.ndarray`
        vector to describe the object
    pose : :obj:`autolab_core.RigidTransform`
        pose of object for the descriptor, if relevant
    """
    __metaclass__ = ABCMeta

    def __init__(self, key, descriptor, pose=None):
        self.key_ = key
        self.descriptor_ = descriptor
        self.pose_ = pose

    @property
    def key(self):
        return self.key_

    @property
    def descriptor(self):
        return self.descriptor_

    @property
    def pose(self):
        return self.pose_

class SHOTFeature(LocalFeature):
    """ Signature of Oriented Histogram (SHOT) features """ 
    def __init__(self, descriptor, rf, point, normal):
        LocalFeature.__init__(self, descriptor, rf, point, normal)

class MVCNNFeature(GlobalFeature):
    """ Multi-View Convolutional Neural Network (MV-CNN) descriptor """ 
    def __init__(self, key, descriptor, pose=None):
        GlobalFeature.__init__(self, key, descriptor, pose)

class BagOfFeatures:
    """ Wrapper for a list of features, created for the sake of future bag-of-words reps.

    Attributes
    ----------
    features : :obj:`list` of :obj:`Feature`
        list of feature objects
    """
    def __init__(self, features = None):
        self.features_ = features
        if self.features_ is None:
            self.features_ = []

        self.num_features_ = len(self.features_)

    def add(self, feature):
        """ Add a new feature to the bag.

        Parameters
        ----------
        feature : :obj:`Feature`
            feature to add
        """
        self.features_.append(feature)
        self.num_features_ = len(self.features_)        

    def extend(self, features):
        """ Add a list of features to the bag.

        Parameters
        ----------
        feature : :obj:`list` of :obj:`Feature`
            features to add
        """
        self.features_.extend(features)
        self.num_features_ = len(self.features_)        

    def feature(self, index):
        """ Returns a feature.

        Parameters
        ----------
        index : int
            index of feature in list

        Returns
        -------
        :obj:`Feature`
        """
        if index < 0 or index >= self.num_features_:
            raise ValueError('Index %d out of range' %(index))
        return self.features_[index]

    def feature_subset(self, indices):
        """ Returns some subset of the features.
        
        Parameters
        ----------
        indices : :obj:`list` of :obj:`int`
            indices of the features in the list

        Returns
        -------
        :obj:`list` of :obj:`Feature`
        """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if not isinstance(indices, list):
            raise ValueError('Can only index with lists')
        return [self.features_[i] for i in indices]

    @property
    def num_features(self):
        return self.num_features_

    @property
    def descriptors(self):
        """ Make a nice array of the descriptors """
        return np.array([f.descriptor for f in self.features_])

    @property
    def reference_frames(self):
        """ Make a nice array of the reference frames """
        return np.array([f.reference_frame for f in self.features_])

    @property
    def keypoints(self):
        """ Make a nice array of the keypoints """
        return np.array([f.keypoint for f in self.features_])

    @property
    def normals(self):
        """ Make a nice array of the normals """
        return np.array([f.normal for f in self.features_])
