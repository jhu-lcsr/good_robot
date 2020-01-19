import struct
import math
import numpy as np
import warnings
import cv2
from scipy import ndimage
import datetime
import os


def mkdir_p(path):
    """Create the specified path on the filesystem like the `mkdir -p` command
    Creates one or more filesystem directory levels as needed,
    and does not return an error if the directory already exists.
    """
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Apply a timestamp to the front of a filename description.
    see: http://stackoverflow.com/a/5215012/99379
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def get_pointcloud(color_img, depth_img, camera_intrinsics):

    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), np.linspace(0,im_h-1,im_h))
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0][2],depth_img/camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1][2],depth_img/camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h*im_w,1)
    cam_pts_y.shape = (im_h*im_w,1)
    cam_pts_z.shape = (im_h*im_w,1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:,:,0]
    rgb_pts_g = color_img[:,:,1]
    rgb_pts_b = color_img[:,:,2]
    rgb_pts_r.shape = (im_h*im_w,1)
    rgb_pts_g.shape = (im_h*im_w,1)
    rgb_pts_b.shape = (im_h*im_w,1)

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution, background_heightmap=None, median_filter_pixels=5):

    if median_filter_pixels > 0:
        depth_img = ndimage.median_filter(depth_img, size=median_filter_pixels)

    # Compute heightmap size
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)
    depth_heightmap = np.zeros(heightmap_size)

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))

    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:,2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]

    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1]), surface_pts[:,2] < workspace_limits[2][1])
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]

    # Create orthographic top-down-view RGB-D depth heightmap
    heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    if median_filter_pixels > 0:
        depth_heightmap = ndimage.median_filter(depth_heightmap, size=median_filter_pixels)
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan
    # subtract out the scene background heights, if available
    if background_heightmap is not None:
        depth_heightmap -= background_heightmap

    # Create orthographic top-down-view RGB-D color heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_r[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[0]]
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[1]]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[2]]
    if median_filter_pixels > 0:
        color_heightmap_r = ndimage.median_filter(color_heightmap_r, size=median_filter_pixels)
        color_heightmap_b = ndimage.median_filter(color_heightmap_b, size=median_filter_pixels)
        color_heightmap_g = ndimage.median_filter(color_heightmap_g, size=median_filter_pixels)
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)


    return color_heightmap, depth_heightmap

def common_sense_action_failure_heuristic(heightmap, heightmap_resolution=0.002, gripper_width=0.12, min_contact_height=0.01, push_length=0.0):
    """ Get heuristic scores for the grasp Q value at various pixels. 0 means our model confidently indicates no progress will be made, 1 means progress may be possible.
    """
    pixels_to_dilate = int(np.ceil((gripper_width + push_length)/heightmap_resolution))
    kernel = np.ones((pixels_to_dilate, pixels_to_dilate), np.uint8)
    object_pixels = (heightmap > min_contact_height).astype(np.uint8)
    contactable_regions = cv2.dilate(object_pixels, kernel, iterations=1)
    return contactable_regions

# Save a 3D point cloud to a binary .ply file
def pcwrite(xyz_pts, filename, rgb_pts=None):
    assert xyz_pts.shape[1] == 3, 'input XYZ points should be an Nx3 matrix'
    if rgb_pts is None:
        rgb_pts = np.ones(xyz_pts.shape).astype(np.uint8)*255
    assert xyz_pts.shape == rgb_pts.shape, 'input RGB colors should be Nx3 matrix and same size as input XYZ points'

    # Write header for .ply file
    pc_file = open(filename, 'wb')
    pc_file.write('ply\n')
    pc_file.write('format binary_little_endian 1.0\n')
    pc_file.write('element vertex %d\n' % xyz_pts.shape[0])
    pc_file.write('property float x\n')
    pc_file.write('property float y\n')
    pc_file.write('property float z\n')
    pc_file.write('property uchar red\n')
    pc_file.write('property uchar green\n')
    pc_file.write('property uchar blue\n')
    pc_file.write('end_header\n')

    # Write 3D points to .ply file
    for i in range(xyz_pts.shape[0]):
        pc_file.write(bytearray(struct.pack("fffccc",xyz_pts[i][0],xyz_pts[i][1],xyz_pts[i][2],rgb_pts[i][0].tostring(),rgb_pts[i][1].tostring(),rgb_pts[i][2].tostring())))
    pc_file.close()


def get_affordance_vis(grasp_affordances, input_images, num_rotations, best_pix_ind):
    vis = None
    for vis_row in range(num_rotations/4):
        tmp_row_vis = None
        for vis_col in range(4):
            rotate_idx = vis_row*4+vis_col
            affordance_vis = grasp_affordances[rotate_idx,:,:]
            affordance_vis[affordance_vis < 0] = 0 # assume probability
            # affordance_vis = np.divide(affordance_vis, np.max(affordance_vis))
            affordance_vis[affordance_vis > 1] = 1 # assume probability
            affordance_vis.shape = (grasp_affordances.shape[1], grasp_affordances.shape[2])
            affordance_vis = cv2.applyColorMap((affordance_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
            input_image_vis = (input_images[rotate_idx,:,:,:]*255).astype(np.uint8)
            input_image_vis = cv2.resize(input_image_vis, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            affordance_vis = (0.5*cv2.cvtColor(input_image_vis, cv2.COLOR_RGB2BGR) + 0.5*affordance_vis).astype(np.uint8)
            if rotate_idx == best_pix_ind[0]:
                affordance_vis = cv2.circle(affordance_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
            if tmp_row_vis is None:
                tmp_row_vis = affordance_vis
            else:
                tmp_row_vis = np.concatenate((tmp_row_vis,affordance_vis), axis=1)
        if vis is None:
            vis = tmp_row_vis
        else:
            vis = np.concatenate((vis,tmp_row_vis), axis=0)

    return vis


def get_difference(color_heightmap, color_space, bg_color_heightmap):

    color_space = np.concatenate((color_space, np.asarray([[0.0, 0.0, 0.0]])), axis=0)
    color_space.shape = (color_space.shape[0], 1, 1, color_space.shape[1])
    color_space = np.tile(color_space, (1, color_heightmap.shape[0], color_heightmap.shape[1], 1))

    # Normalize color heightmaps
    color_heightmap = color_heightmap.astype(float)/255.0
    color_heightmap.shape = (1, color_heightmap.shape[0], color_heightmap.shape[1], color_heightmap.shape[2])
    color_heightmap = np.tile(color_heightmap, (color_space.shape[0], 1, 1, 1))

    bg_color_heightmap = bg_color_heightmap.astype(float)/255.0
    bg_color_heightmap.shape = (1, bg_color_heightmap.shape[0], bg_color_heightmap.shape[1], bg_color_heightmap.shape[2])
    bg_color_heightmap = np.tile(bg_color_heightmap, (color_space.shape[0], 1, 1, 1))

    # Compute nearest neighbor distances to key colors
    key_color_dist = np.sqrt(np.sum(np.power(color_heightmap - color_space,2), axis=3))
    # key_color_dist_prob = F.softmax(Variable(torch.from_numpy(key_color_dist), volatile=True), dim=0).data.numpy()

    bg_key_color_dist = np.sqrt(np.sum(np.power(bg_color_heightmap - color_space,2), axis=3))
    # bg_key_color_dist_prob = F.softmax(Variable(torch.from_numpy(bg_key_color_dist), volatile=True), dim=0).data.numpy()

    key_color_match = np.argmin(key_color_dist, axis=0)
    bg_key_color_match = np.argmin(bg_key_color_dist, axis=0)
    key_color_match[key_color_match == color_space.shape[0] - 1] = color_space.shape[0] + 1
    bg_key_color_match[bg_key_color_match == color_space.shape[0] - 1] = color_space.shape[0] + 2

    return np.sum(key_color_match == bg_key_color_match).astype(float)/np.sum(bg_key_color_match < color_space.shape[0]).astype(float)


# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


# Checks if a matrix is a valid rotation matrix.
def isRotm(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotm2euler(R) :

    assert(isRotm(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def angle2rotm(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke

    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis_magnitude = np.linalg.norm(axis)
    axis = np.divide(axis, axis_magnitude, out=np.zeros_like(axis), where=axis_magnitude!=0)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.array(np.outer(axis, axis) * (1.0 - cosa))
    axis *= sina
    RA = np.array([[ 0.0,     -axis[2],  axis[1]],
                      [ axis[2], 0.0,      -axis[0]],
                      [-axis[1], axis[0],  0.0]])
    R = RA + np.array(R)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:

        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotm2angle(R):
    # From: euclideanspace.com

    epsilon = 0.01 # Margin to allow for rounding errors
    epsilon2 = 0.1 # Margin to distinguish between 0 and 180 degrees

    assert(isRotm(R))

    if ((abs(R[0][1]-R[1][0])< epsilon) and (abs(R[0][2]-R[2][0])< epsilon) and (abs(R[1][2]-R[2][1])< epsilon)):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if ((abs(R[0][1]+R[1][0]) < epsilon2) and (abs(R[0][2]+R[2][0]) < epsilon2) and (abs(R[1][2]+R[2][1]) < epsilon2) and (abs(R[0][0]+R[1][1]+R[2][2]-3) < epsilon2)):
            # this singularity is identity matrix so angle = 0
            return [0,1,0,0] # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0]+1)/2
        yy = (R[1][1]+1)/2
        zz = (R[2][2]+1)/2
        xy = (R[0][1]+R[1][0])/4
        xz = (R[0][2]+R[2][0])/4
        yz = (R[1][2]+R[2][1])/4
        if ((xx > yy) and (xx > zz)): # R[0][0] is the largest diagonal term
            if (xx< epsilon):
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy/x
                z = xz/x
        elif (yy > zz): # R[1][1] is the largest diagonal term
            if (yy< epsilon):
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy/y
                z = yz/y
        else: # R[2][2] is the largest diagonal term so base result on this
            if (zz< epsilon):
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz/z
                y = yz/z
        return [angle,x,y,z] # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt((R[2][1] - R[1][2])*(R[2][1] - R[1][2]) + (R[0][2] - R[2][0])*(R[0][2] - R[2][0]) + (R[1][0] - R[0][1])*(R[1][0] - R[0][1])) # used to normalise
    if (abs(s) < 0.001):
        s = 1

    # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos(( R[0][0] + R[1][1] + R[2][2] - 1)/2)
    x = (R[2][1] - R[1][2])/s
    y = (R[0][2] - R[2][0])/s
    z = (R[1][0] - R[0][1])/s
    return [angle,x,y,z]


def quat2rotm(quat):
    """
    Quaternion to rotation matrix.

    Args:
    - quat (4, numpy array): quaternion w, x, y, z
    Returns:
    - rotm: (3x3 numpy array): rotation matrix
    """
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    s = w*w + x*x + y*y + z*z

    rotm = np.array([[1-2*(y*y+z*z)/s, 2*(x*y-z*w)/s,   2*(x*z+y*w)/s  ],
                     [2*(x*y+z*w)/s,   1-2*(x*x+z*z)/s, 2*(y*z-x*w)/s  ],
                     [2*(x*z-y*w)/s,   2*(y*z+x*w)/s,   1-2*(x*x+y*y)/s]
    ])

    return rotm


def make_rigid_transformation(pos, orn):
    """
    Rigid transformation from position and orientation.
    Args:
    - pos (3, numpy array): translation
    - orn (4, numpy array): orientation in quaternion
    Returns:
    - homo_mat (4x4 numpy array): homogenenous transformation matrix
    """
    rotm = quat2rotm(orn)
    homo_mat = np.c_[rotm, np.reshape(pos, (3, 1))]
    homo_mat = np.r_[homo_mat, [[0, 0, 0, 1]]]

    return homo_mat


def axis_angle_and_translation_to_rigid_transformation(tool_position, tool_orientation):
    tool_orientation_angle = np.linalg.norm(tool_orientation)
    tool_orientation_axis = tool_orientation/tool_orientation_angle
    # Note that this following rotm is the base frame in tool frame
    tool_orientation_rotm = angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3,:3]
    # Tool rigid body transformation
    tool_transformation = np.zeros((4, 4))
    tool_transformation[:3, :3] = tool_orientation_rotm
    tool_transformation[:3, 3] = tool_position
    tool_transformation[3, 3] = 1
    return tool_transformation


def axxb(robotPose, markerPose, baseToCamera=True):
    """
    Copyright (c) 2019, Hongtao Wu
    AX=XB solver for eye-on base
    Using the Park and Martin Method: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=326576

    Args:
    - robotPose (list of 4x4 numpy array): poses (homogenous transformation) of the robot end-effector in the robot base frame.
    - markerPose (list of 4x4 numpy array): poses (homogenous transformation) of the marker in the camera frame.
    - baseToCamera (boolean): If true it will compute the base to camera transform, if false it will compute the robot tip to fiducial transform.

    Return:
    - cam2base (4x4 numpy array): poses of the camera in robot base frame.
    """

    assert len(robotPose) == len(markerPose), 'robot poses and marker poses are not of the same length!'

    n = len(robotPose)
    print("Total number of poses: %i" % n)
    A = np.zeros((4, 4, n-1))
    B = np.zeros((4, 4, n-1))
    alpha = np.zeros((3, n-1))
    beta = np.zeros((3, n-1))

    M = np.zeros((3, 3))

    nan_num = 0

    sequence = np.arange(n)
    np.random.shuffle(sequence)

    for i in range(n-1):
        if baseToCamera:
            # compute the robot base to the robot camera
            A[:, :, i] = np.matmul(robotPose[sequence[i+1]], pose_inv(robotPose[sequence[i]]))
            B[:, :, i] = np.matmul(markerPose[sequence[i+1]], pose_inv(markerPose[sequence[i]]))
        else:
            # compute the robot tool tip to the robot fiducial marker seen by the camera
            A[:, :, i] = np.matmul(pose_inv(robotPose[sequence[i+1]]), robotPose[sequence[i]])
            B[:, :, i] = np.matmul(pose_inv(markerPose[sequence[i+1]]), markerPose[sequence[i]])

        alpha[:, i] = get_mat_log(A[:3, :3, i])
        beta[:, i] = get_mat_log(B[:3, :3, i])

        # Bad pair of transformation are very close in the orientation.
        # They will give nan result
        if np.sum(np.isnan(alpha[:, i])) + np.sum(np.isnan(beta[:, i])) > 0:
            nan_num += 1
            continue
        else:
            M += np.outer(beta[:, i], alpha[:, i])

    print("Invalid poses number: {}".format(nan_num))

    # Get the rotation matrix
    mtm = np.matmul(M.T, M)
    u_mtm, s_mtm, vh_mtm = np.linalg.svd(mtm)

    R = np.matmul(np.matmul(np.matmul(u_mtm, np.diag(np.power(s_mtm, -0.5))), vh_mtm), M.T)

    # Get the tranlation vector
    I_Ra_Left = np.zeros((3*(n-1), 3))
    ta_Rtb_Right = np.zeros((3 * (n-1), 1))
    for i in range(n-1):
        I_Ra_Left[(3*i):(3*(i+1)), :] = np.eye(3) - A[:3, :3, i]
        ta_Rtb_Right[(3*i):(3*(i+1)), :] = np.reshape(A[:3, 3, i] - np.dot(R, B[:3, 3, i]), (3, 1))
    t = np.linalg.lstsq(I_Ra_Left, ta_Rtb_Right, rcond=None)[0]

    cam2base = np.c_[R, t]
    cam2base = np.r_[cam2base, [[0, 0, 0, 1]]]

    return cam2base


def pose_inv(pose):
    """
    Inverse of a homogenenous transformation.
    Args:
    - pose (4x4 numpy array)
    Return:
    - inv_pose (4x4 numpy array)
    """
    R = pose[:3, :3]
    t = pose[:3, 3]

    inv_R = R.T
    inv_t = - np.dot(inv_R, t)

    inv_pose = np.c_[inv_R, np.transpose(inv_t)]
    inv_pose = np.r_[inv_pose, [[0, 0, 0, 1]]]

    return inv_pose


def get_mat_log(R):
    """
    Get the log(R) of the rotation matrix R.

    Args:
    - R (3x3 numpy array): rotation matrix
    Returns:
    - w (3, numpy array): log(R)
    """
    theta = np.arccos((np.trace(R) - 1) / 2)
    w_hat = (R - R.T) * theta / (2 * np.sin(theta))  # Skew symmetric matrix
    w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])  # [w1, w2, w3]

    return w


def calib_grid_cartesian(workspace_limits, calib_grid_step):
    """
    Construct 3D calibration grid across workspace

    # Arguments

        workspace_limits: list of [min,max] coordinates for the list [x, y, z] in meters.
        calib_grid_step: the step size of points in a 3d grid to be created in meters.

    # Returns

        num_calib_grid_pts, calib_grid_pts
    """
    gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], (workspace_limits[0][1] - workspace_limits[0][0])/calib_grid_step)
    gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], (workspace_limits[1][1] - workspace_limits[1][0])/calib_grid_step)
    gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], (workspace_limits[2][1] - workspace_limits[2][0])/calib_grid_step)
    calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
    num_calib_grid_pts = calib_grid_x.shape[0]*calib_grid_x.shape[1]*calib_grid_x.shape[2]
    calib_grid_x.shape = (num_calib_grid_pts,1)
    calib_grid_y.shape = (num_calib_grid_pts,1)
    calib_grid_z.shape = (num_calib_grid_pts,1)
    calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)
    return num_calib_grid_pts, calib_grid_pts


def check_separation(values, distance_threshold):
    """Checks that the separation among the values is close enough about distance_threshold.

    :param values: array of values to check, assumed to be sorted from low to high
    :param distance_threshold: threshold
    :returns: success
    :rtype: bool

    """
    for i in range(len(values) - 1):
        x = values[i]
        y = values[i + 1]
        assert x < y, '`values` assumed to be sorted'
        if y < x + distance_threshold / 2.:
            # print('check_separation(): not long enough for idx: {}'.format(i))
            return False
        if y - x > distance_threshold:
            # print('check_separation(): too far apart')
            return False
    return True


def polyfit(*args, **kwargs):
    with warnings.catch_warnings():
        # suppress the RankWarning, which just means the best fit line was bad.
        warnings.simplefilter('ignore', np.RankWarning)
        out = np.polyfit(*args, **kwargs)
    return out
