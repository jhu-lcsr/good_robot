"""
Classes for feature matching between point sets for registration
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import spatial
import scipy.spatial.distance as ssd

from autolab_core import PointCloud

from .features import BagOfFeatures

class Correspondences:
    """ Wrapper for point-set correspondences.

    Attributes
    ----------
    index_map : :obj:`list` of int
        maps list indices (source points) to target point indices
    source_points : Nx3 :obj:`numpy.ndarray`
        set of source points for registration
    target_points : Nx3 :obj:`numpy.ndarray`
        set of target points for registration
    num_matches : int
        the total number of matches
    """
    def __init__(self, index_map, source_points, target_points):
        self.index_map_ = index_map
        self.source_points_ = source_points
        self.target_points_ = target_points
        self.num_matches_ = source_points.shape[0]

    @property
    def index_map(self):
        return self.index_map_

    @property
    def source_points(self):
        return self.source_points_

    @property
    def target_points(self):
        return self.target_points_

    @property
    def num_matches(self):
        return self.num_matches_

    # Functions to iterate through matches like "for source_corr, target_corr in correspondences"
    def __iter__(self):
        self.iter_count_ = 0

    def next(self):
        if self.iter_count_ >= len(self.num_matches_):
            raise StopIteration
        else:
            return self.source_points_[self.iter_count,:], self.target_points_[self.iter_count,:]

class NormalCorrespondences(Correspondences):
    """ Wrapper for point-set correspondences with surface normals.

    Attributes
    ----------
    index_map : :obj:`list` of int
        maps list indices (source points) to target point indices
    source_points : Nx3 :obj:`numpy.ndarray`
        set of source points for registration
    target_points : Nx3 :obj:`numpy.ndarray`
        set of target points for registration
    source_normals : normalized Nx3 :obj:`numpy.ndarray`
        set of source normals for registration
    target_normals : normalized Nx3 :obj:`numpy.ndarray`
        set of target points for registration
    num_matches : int
        the total number of matches
    """
    def __init__(self, index_map, source_points, target_points, source_normals, target_normals):
        self.source_normals_ = source_normals
        self.target_normals_ = target_normals
        Correspondences.__init__(self, index_map, source_points, target_points)

    @property
    def source_normals(self):
        return self.source_normals_

    @property
    def target_normals(self):
        return self.target_normals_

    # Functions to iterate through matches like "for source_corr, target_corr in correspondences"
    def __iter__(self):
        self.iter_count_ = 0

    def next(self):
        if self.iter_count_ >= len(self.num_matches_):
            raise StopIteration
        else:
            return self.source_points_[self.iter_count,:], self.target_points_[self.iter_count,:], self.source_normals_[self.iter_count,:], self.target_normals_[self.iter_count,:]

class FeatureMatcher:
    """
    Generic feature matching between local features on a source and target object using nearest neighbors.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @staticmethod
    def get_point_index(point, all_points, eps = 1e-4):
        """ Get the index of a point in an array """
        inds = np.where(np.linalg.norm(point - all_points, axis=1) < eps)
        if inds[0].shape[0] == 0:
            return -1
        return inds[0][0]

    @abstractmethod
    def match(self, source_obj, target_obj):
        """
        Matches features between a source and target object. Source and target object types depend on subclass implementation.
        """
        pass

class RawDistanceFeatureMatcher(FeatureMatcher):
    def match(self, source_obj_features, target_obj_features):
        """
        Matches features between two graspable objects based on a full distance matrix.

        Parameters
        ----------
        source_obj_features : :obj:`BagOfFeatures`
            bag of the source objects features
        target_obj_features : :obj:`BagOfFeatures`
            bag of the target objects features

        Returns
        -------
        corrs : :obj:`Correspondences`
            the correspondences between source and target
        """
        if not isinstance(source_obj_features, f.BagOfFeatures):
            raise ValueError('Must supply source bag of object features')
        if not isinstance(target_obj_features, f.BagOfFeatures):
            raise ValueError('Must supply target bag of object features')

        # source feature descriptors and keypoints
        source_descriptors = source_obj_features.descriptors
        target_descriptors = target_obj_features.descriptors
        source_keypoints = source_obj_features.keypoints
        target_keypoints = target_obj_features.keypoints

        #calculate distance between this model's descriptors and each of the other_model's descriptors
        dists = spatial.distance.cdist(source_descriptors, target_descriptors)

        #calculate the indices of the target_model that minimize the distance to the descriptors in this model
        source_closest_descriptors = dists.argmin(axis=1)
        target_closest_descriptors = dists.argmin(axis=0)
        match_indices = []
        source_matched_points = np.zeros((0,3))
        target_matched_points = np.zeros((0,3))

        #calculate which points/indices the closest descriptors correspond to
        for i, j in enumerate(source_closest_descriptors):
            # for now, only keep correspondences that are a 2-way match
            if target_closest_descriptors[j] == i:
                match_indices.append(j)
                source_matched_points = np.r_[source_matched_points, source_keypoints[i:i+1, :]]
                target_matched_points = np.r_[target_matched_points, target_keypoints[j:j+1, :]]
            else:
                match_indices.append(-1)

        return Correspondences(match_indices, source_matched_points, target_matched_points)

class PointToPlaneFeatureMatcher(FeatureMatcher):
    """ Match points using a point to plane criterion with thresholding.

    Attributes
    ----------
    dist_thresh : float
        threshold distance to consider a match valid
    norm_thresh : float
        threshold cosine distance alignment betwen normals to consider a match valid
    """
    def __init__(self, dist_thresh=0.05, norm_thresh=0.75):
        self.dist_thresh_ = dist_thresh
        self.norm_thresh_ = norm_thresh
        FeatureMatcher.__init__(self)

    def match(self, source_points, target_points, source_normals, target_normals):
        """
        Matches points between two point-normal sets. Uses the closest ip to choose matches, with distance for thresholding only.

        Parameters
        ----------
        source_point_cloud : Nx3 :obj:`numpy.ndarray`
            source object points
        target_point_cloud : Nx3 :obj:`numpy.ndarray`
            target object points
        source_normal_cloud : Nx3 :obj:`numpy.ndarray`
            source object outward-pointing normals
        target_normal_cloud : Nx3 :obj`numpy.ndarray`
            target object outward-pointing normals

        Returns
        -------
        :obj`Correspondences`
            the correspondences between source and target
        """
        # compute the distances and inner products between the point sets
        dists = ssd.cdist(source_points, target_points, 'euclidean')
        ip = source_normals.dot(target_normals.T) # abs because we don't have correct orientations
        source_ip = source_points.dot(target_normals.T)
        target_ip = target_points.dot(target_normals.T)
        target_ip = np.diag(target_ip)
        target_ip = np.tile(target_ip, [source_points.shape[0], 1])
        abs_diff = np.abs(source_ip - target_ip) # difference in inner products

        # mark invalid correspondences
        invalid_dists = np.where(dists > self.dist_thresh_)
        abs_diff[invalid_dists[0], invalid_dists[1]] = np.inf
        invalid_norms = np.where(ip < self.norm_thresh_)
        abs_diff[invalid_norms[0], invalid_norms[1]] = np.inf

        # choose the closest matches
        match_indices = np.argmin(abs_diff, axis=1)
        match_vals = np.min(abs_diff, axis=1)
        invalid_matches = np.where(match_vals == np.inf)
        match_indices[invalid_matches[0]] = -1

        return NormalCorrespondences(match_indices, source_points, target_points, source_normals, target_normals)
