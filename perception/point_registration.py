"""
Classes for point set registration using variants of Iterated-Closest Point
Author: Jeff Mahler
"""
from abc import ABCMeta, abstractmethod
import logging
import numpy as np

from autolab_core import RigidTransform, PointCloud, NormalCloud, skew

from .feature_matcher import PointToPlaneFeatureMatcher

class RegistrationResult(object):
    """ Struct to hold results of point set registration.

    Attributes
    ----------
    T_source_target : :obj:`autolab_core.RigidTranform`
        transformation from source to target frame
    cost : float
        numeric value of the registration objective for the given transform
    """
    def __init__(self, T_source_target, cost):
        self.T_source_target = T_source_target
        self.cost = cost

class IterativeRegistrationSolver:
    """ Abstract class for iterative registration solvers. """
    __metaclass__ = ABCMeta

    @abstractmethod
    def register(self, source_point_cloud, target_point_cloud,
                 source_normal_cloud, target_normal_cloud, matcher,
                 num_iterations=1, compute_total_cost=True, match_centroids=False,
                 vis=False):
        """ Iteratively register objects to one another.

        Parameters
        ----------
        source_point_cloud : :obj:`autolab_core.PointCloud`
            source object points
        target_point_cloud : :obj`autolab_core.PointCloud`
            target object points
        source_normal_cloud : :obj:`autolab_core.NormalCloud`
            source object outward-pointing normals
        target_normal_cloud : :obj:`autolab_core.NormalCloud`
            target object outward-pointing normals
        matcher : :obj:`PointToPlaneFeatureMatcher`
            object to match the point sets
        num_iterations : int
            the number of iterations to run
        compute_total_cost : bool
            whether or not to compute the total cost upon termination.
        match_centroids : bool
            whether or not to match the centroids of the point clouds
        
        Returns
        -------
        :obj`RegistrationResult`
            results containing source to target transformation and cost
        """
        pass

class PointToPlaneICPSolver(IterativeRegistrationSolver):
    """ Performs Iterated Closest Point with an objective weighted between point-to-point and point-to-plane.

    Attributes
    ----------
    sample_size : int
        number of randomly sampled points to use per iteration
    cost_sample_size : int
        number of randomly sampled points to use for cost evaluations
    gamma : float
        weight of point-to-point objective relative to point-to-plane objective
    mu : float
        regularizer for matrix inversion in the Gauss-Newton step
    """
    def __init__(self, sample_size=100, cost_sample_size=100, gamma=100.0, mu=1e-2):
        self.sample_size_ = sample_size
        self.cost_sample_size_ = cost_sample_size
        self.gamma_ = gamma
        self.mu_ = mu
        IterativeRegistrationSolver.__init__(self)
    
    def register(self, source_point_cloud, target_point_cloud,
                 source_normal_cloud, target_normal_cloud, matcher,
                 num_iterations=1, compute_total_cost=True, match_centroids=False,
                 vis=False):
        """
        Iteratively register objects to one another using a modified version of point to plane ICP.
        The cost func is PointToPlane_COST + gamma * PointToPoint_COST.
        Uses a `stochastic Gauss-Newton step` where on each iteration a smaller number of points is sampled.

        Parameters
        ----------
        source_point_cloud : :obj:`autolab_core.PointCloud`
            source object points
        target_point_cloud : :obj`autolab_core.PointCloud`
            target object points
        source_normal_cloud : :obj:`autolab_core.NormalCloud`
            source object outward-pointing normals
        target_normal_cloud : :obj:`autolab_core.NormalCloud`
            target object outward-pointing normals
        matcher : :obj:`PointToPlaneFeatureMatcher`
            object to match the point sets
        num_iterations : int
            the number of iterations to run
        compute_total_cost : bool
            whether or not to compute the total cost upon termination.
        match_centroids : bool
            whether or not to match the centroids of the point clouds
        
        Returns
        -------
        :obj`RegistrationResult`
            results containing source to target transformation and cost
        """
        # check valid data
        if not isinstance(source_point_cloud, PointCloud) or not isinstance(target_point_cloud, PointCloud):
            raise ValueError('Source and target point clouds must be PointCloud objects')
        if not isinstance(source_normal_cloud, NormalCloud) or not isinstance(target_normal_cloud, NormalCloud):
            raise ValueError('Source and target normal clouds must be NormalCloud objects')
        if not isinstance(matcher, PointToPlaneFeatureMatcher):
            raise ValueError('Feature matcher must be a PointToPlaneFeatureMatcher object')
        if source_point_cloud.num_points != source_normal_cloud.num_points or target_point_cloud.num_points != target_normal_cloud.num_points:
            raise ValueError('Input point clouds must have the same number of points as corresponding normal cloud')

        # extract source and target point and normal data arrays
        orig_source_points = source_point_cloud.data.T
        orig_target_points = target_point_cloud.data.T
        orig_source_normals = source_normal_cloud.data.T
        orig_target_normals = target_normal_cloud.data.T

        # setup the problem
        normal_norms = np.linalg.norm(orig_target_normals, axis=1)
        valid_inds = np.nonzero(normal_norms)
        orig_target_points = orig_target_points[valid_inds[0],:]
        orig_target_normals = orig_target_normals[valid_inds[0],:]

        normal_norms = np.linalg.norm(orig_source_normals, axis=1)
        valid_inds = np.nonzero(normal_norms)
        orig_source_points = orig_source_points[valid_inds[0],:]
        orig_source_normals = orig_source_normals[valid_inds[0],:]

        # alloc buffers for solutions
        source_mean_point = np.mean(orig_source_points, axis=0)
        target_mean_point = np.mean(orig_target_points, axis=0)
        R_sol = np.eye(3)
        t_sol = np.zeros([3, 1]) #init with diff between means
        if match_centroids:
            t_sol[:,0] = target_mean_point - source_mean_point

        # iterate through
        for i in range(num_iterations):
            logging.info('Point to plane ICP iteration %d' %(i))

            # subsample points
            source_subsample_inds = np.random.choice(orig_source_points.shape[0], size=self.sample_size_)
            source_points = orig_source_points[source_subsample_inds,:]
            source_normals = orig_source_normals[source_subsample_inds,:]
            target_subsample_inds = np.random.choice(orig_target_points.shape[0], size=self.sample_size_)
            target_points = orig_target_points[target_subsample_inds,:]
            target_normals = orig_target_normals[target_subsample_inds,:]

            # transform source points
            source_points = (R_sol.dot(source_points.T) + np.tile(t_sol, [1, source_points.shape[0]])).T
            source_normals = (R_sol.dot(source_normals.T)).T
        
            # closest points
            corrs = matcher.match(source_points, target_points, source_normals, target_normals)

            # solve optimal rotation + translation
            valid_corrs = np.where(corrs.index_map != -1)[0]
            source_corr_points = corrs.source_points[valid_corrs,:]
            target_corr_points = corrs.target_points[corrs.index_map[valid_corrs], :]
            target_corr_normals = corrs.target_normals[corrs.index_map[valid_corrs], :]

            num_corrs = valid_corrs.shape[0]
            if num_corrs == 0:
                logging.warning('No correspondences found')
                break

            # create A and b matrices for Gauss-Newton step on joint cost function
            A = np.zeros([6,6])
            b = np.zeros([6,1])
            Ap = np.zeros([6,6])
            bp = np.zeros([6,1])
            G = np.zeros([3,6])
            G[:,3:] = np.eye(3)

            for i in range(num_corrs):
                s = source_corr_points[i:i+1,:].T
                t = target_corr_points[i:i+1,:].T
                n = target_corr_normals[i:i+1,:].T
                G[:,:3] = skew(s).T
                A += G.T.dot(n).dot(n.T).dot(G)
                b += G.T.dot(n).dot(n.T).dot(t - s)

                Ap += G.T.dot(G)
                bp += G.T.dot(t - s)
            v = np.linalg.solve(A + self.gamma_*Ap + self.mu_*np.eye(6),
                                b + self.gamma_*bp)

            # create pose values from the solution
            R = np.eye(3)
            R = R + skew(v[:3])
            U, S, V = np.linalg.svd(R)
            R = U.dot(V)
            t = v[3:]

            # incrementally update the final transform
            R_sol = R.dot(R_sol)
            t_sol = R.dot(t_sol) + t

        T_source_target = RigidTransform(R_sol, t_sol, from_frame=source_point_cloud.frame, to_frame=target_point_cloud.frame)

        total_cost = 0
        source_points = (R_sol.dot(orig_source_points.T) + np.tile(t_sol, [1, orig_source_points.shape[0]])).T
        source_normals = (R_sol.dot(orig_source_normals.T)).T

        if compute_total_cost:
            # rematch all points to get the final cost
            corrs = matcher.match(source_points, orig_target_points, source_normals, orig_target_normals)
            valid_corrs = np.where(corrs.index_map != -1)[0]
            num_corrs = valid_corrs.shape[0]
            if num_corrs == 0:
                return RegistrationResult(T_source_target, np.inf)

            # get the corresponding points
            source_corr_points = corrs.source_points[valid_corrs,:]
            target_corr_points = corrs.target_points[corrs.index_map[valid_corrs], :]
            target_corr_normals = corrs.target_normals[corrs.index_map[valid_corrs], :]

            # determine total cost
            source_target_alignment = np.diag((source_corr_points - target_corr_points).dot(target_corr_normals.T))
            point_plane_cost = (1.0 / num_corrs) * np.sum(source_target_alignment * source_target_alignment)
            point_dist_cost = (1.0 / num_corrs) * np.sum(np.linalg.norm(source_corr_points - target_corr_points, axis=1)**2)
            total_cost = point_plane_cost + self.gamma_ * point_dist_cost

        return RegistrationResult(T_source_target, total_cost)

    def register_2d(self, source_point_cloud, target_point_cloud,
                    source_normal_cloud, target_normal_cloud, matcher,
                    num_iterations=1, compute_total_cost=True, vis=False):
        """
        Iteratively register objects to one another using a modified version of point to plane ICP
        which only solves for tx and ty (translation in the plane) and theta (rotation about the z axis).
        The cost func is actually PointToPlane_COST + gamma * PointToPoint_COST
        Points should be specified in the basis of the planar worksurface.

        Parameters
        ----------
        source_point_cloud : :obj:`autolab_core.PointCloud`
            source object points
        target_point_cloud : :obj`autolab_core.PointCloud`
            target object points
        source_normal_cloud : :obj:`autolab_core.NormalCloud`
            source object outward-pointing normals
        target_normal_cloud : :obj:`autolab_core.NormalCloud`
            target object outward-pointing normals
        matcher : :obj:`PointToPlaneFeatureMatcher`
            object to match the point sets
        num_iterations : int
            the number of iterations to run
        compute_total_cost : bool
            whether or not to compute the total cost upon termination.
        
        Returns
        -------
        :obj`RegistrationResult`
            results containing source to target transformation and cost
        """     
        if not isinstance(source_point_cloud, PointCloud) or not isinstance(target_point_cloud, PointCloud):
            raise ValueError('Source and target point clouds must be PointCloud objects')
        if not isinstance(source_normal_cloud, NormalCloud) or not isinstance(target_normal_cloud, NormalCloud):
            raise ValueError('Source and target normal clouds must be NormalCloud objects')
        if not isinstance(matcher, PointToPlaneFeatureMatcher):
            raise ValueError('Feature matcher must be a PointToPlaneFeatureMatcher object')
        if source_point_cloud.num_points != source_normal_cloud.num_points or target_point_cloud.num_points != target_normal_cloud.num_points:
            raise ValueError('Input point clouds must have the same number of points as corresponding normal cloud')

        # extract source and target point and normal data arrays
        orig_source_points = source_point_cloud.data.T
        orig_target_points = target_point_cloud.data.T
        orig_source_normals = source_normal_cloud.data.T
        orig_target_normals = target_normal_cloud.data.T

        # setup the problem
        logging.info('Setting up problem')
        normal_norms = np.linalg.norm(orig_target_normals, axis=1)
        valid_inds = np.nonzero(normal_norms)
        orig_target_points = orig_target_points[valid_inds[0],:]
        orig_target_normals = orig_target_normals[valid_inds[0],:]

        normal_norms = np.linalg.norm(orig_source_normals, axis=1)
        valid_inds = np.nonzero(normal_norms)
        orig_source_points = orig_source_points[valid_inds[0],:]
        orig_source_normals = orig_source_normals[valid_inds[0],:]

        # alloc buffers for solutions
        source_mean_point = np.mean(orig_source_points, axis=0)
        target_mean_point = np.mean(orig_target_points, axis=0)
        R_sol = np.eye(3)
        t_sol = np.zeros([3, 1])

        # iterate through
        for i in range(num_iterations):
            logging.info('Point to plane ICP iteration %d' %(i))

            # subsample points
            source_subsample_inds = np.random.choice(orig_source_points.shape[0], size=self.sample_size_)
            source_points = orig_source_points[source_subsample_inds,:]
            source_normals = orig_source_normals[source_subsample_inds,:]
            target_subsample_inds = np.random.choice(orig_target_points.shape[0], size=self.sample_size_)
            target_points = orig_target_points[target_subsample_inds,:]
            target_normals = orig_target_normals[target_subsample_inds,:]

            # transform source points
            source_points = (R_sol.dot(source_points.T) + np.tile(t_sol, [1, source_points.shape[0]])).T
            source_normals = (R_sol.dot(source_normals.T)).T
        
            # closest points
            corrs = matcher.match(source_points, target_points, source_normals, target_normals)

            # solve optimal rotation + translation
            valid_corrs = np.where(corrs.index_map != -1)[0]
            source_corr_points = corrs.source_points[valid_corrs,:]
            target_corr_points = corrs.target_points[corrs.index_map[valid_corrs], :]
            target_corr_normals = corrs.target_normals[corrs.index_map[valid_corrs], :]

            num_corrs = valid_corrs.shape[0]
            if num_corrs == 0:
                break

            # create A and b matrices for Gauss-Newton step on joint cost function
            A = np.zeros([3,3]) # A and b for point to plane cost
            b = np.zeros([3,1])
            Ap = np.zeros([3,3]) # A and b for point to point cost
            bp = np.zeros([3,1])
            G = np.zeros([3,3])
            G[:2,1:] = np.eye(2)

            for i in range(num_corrs):
                s = source_corr_points[i:i+1,:].T
                t = target_corr_points[i:i+1,:].T
                n = target_corr_normals[i:i+1,:].T
                G[0,0] = -s[1] 
                G[1,0] = s[0]
                A += G.T.dot(n).dot(n.T).dot(G)
                b += G.T.dot(n).dot(n.T).dot(t - s)

                Ap += G.T.dot(G)
                bp += G.T.dot(t - s)
            v = np.linalg.solve(A + self.gamma_*Ap + self.mu_*np.eye(3),
                                b + self.gamma_*bp)

            # create pose values from the solution
            R = np.eye(3)
            R = R + skew(np.array([[0],[0],[v[0,0]]]))
            U, S, V = np.linalg.svd(R)
            R = U.dot(V)
            t = np.array([[v[1,0]], [v[2,0]], [0]])

            # incrementally update the final transform
            R_sol = R.dot(R_sol)
            t_sol = R.dot(t_sol) + t

        # compute solution transform
        T_source_target = RigidTransform(R_sol, t_sol, from_frame=source_point_cloud.frame, to_frame=target_point_cloud.frame)

        total_cost = 0
        if compute_total_cost:
            # subsample points
            source_subsample_inds = np.random.choice(orig_source_points.shape[0], size=self.cost_sample_size_)
            source_points = orig_source_points[source_subsample_inds,:]
            source_normals = orig_source_normals[source_subsample_inds,:]
            target_subsample_inds = np.random.choice(orig_target_points.shape[0], size=self.cost_sample_size_)
            target_points = orig_target_points[target_subsample_inds,:]
            target_normals = orig_target_normals[target_subsample_inds,:]

            # transform source points
            source_points = (R_sol.dot(source_points.T) + np.tile(t_sol, [1, source_points.shape[0]])).T
            source_normals = (R_sol.dot(source_normals.T)).T

            # rematch to get the total cost
            corrs = matcher.match(source_points, target_points, source_normals, target_normals)
            valid_corrs = np.where(corrs.index_map != -1)[0]
            num_corrs = valid_corrs.shape[0]
            if num_corrs == 0:
                return RegistrationResult(T_source_target, np.inf)

            # get the corresponding points
            source_corr_points = corrs.source_points[valid_corrs,:]
            target_corr_points = corrs.target_points[corrs.index_map[valid_corrs], :]
            target_corr_normals = corrs.target_normals[corrs.index_map[valid_corrs], :]

            # determine total cost
            source_target_alignment = np.diag((source_corr_points - target_corr_points).dot(target_corr_normals.T))
            point_plane_cost = (1.0 / num_corrs) * np.sum(source_target_alignment * source_target_alignment)
            point_dist_cost = (1.0 / num_corrs) * np.sum(np.linalg.norm(source_corr_points - target_corr_points, axis=1)**2)
            total_cost = point_plane_cost + self.gamma_ * point_dist_cost

        return RegistrationResult(T_source_target, total_cost)

