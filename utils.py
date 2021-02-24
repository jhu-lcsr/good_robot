import struct
import math
import numpy as np
import warnings
import cv2
from scipy import ndimage
import datetime
import os
import json
import yaml
import torch
from scipy.special import softmax
import pathlib
import matplotlib.pyplot as plt

# Import necessary packages
#try:
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from language_embedders import RandomEmbedder, GloveEmbedder, BERTEmbedder
from transformer import TransformerEncoder
from train_language_encoder import get_free_gpu, load_data, get_vocab, LanguageTrainer, FlatLanguageTrainer
#except ImportError:
#    print('Unable to import the language embedder, language trainer, or transformer encoder. This is OK if you are not using the language model.')

# to convert action names to the corresponding ID number and vice-versa
ACTION_TO_ID = {'push': 0, 'grasp': 1, 'place': 2}
ID_TO_ACTION = {0: 'push', 1: 'grasp', 2: 'place'}


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


class NumpyEncoder(json.JSONEncoder):
    """ json encoder for numpy types
    source: https://stackoverflow.com/a/49677241/99379
    """
    def default(self, obj):
        if isinstance(obj,
            (np.int_, np.intc, np.intp, np.int8,
             np.int16, np.int32, np.int64, np.uint8,
             np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj,
           (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def clearance_log_to_trial_count(clearance_log):
    """ Convert clearance log list of end indices to a list of the current trial number at each iteration.

    # Returns

    List of lists of the current trial index.
    ex: [[0], [0], [0], [1], [1]]
    """
    if not len(clearance_log):
        return []
    clearance_log = np.squeeze(clearance_log).astype(np.int)
    # Make a list of the right length containing all zeros
    trial_count = []
    prev_trial_end_index = 0
    for trial_num, trial_end_index in enumerate(clearance_log):
        trial_count += [[trial_num]] * int(trial_end_index - prev_trial_end_index)
        prev_trial_end_index = trial_end_index
    return trial_count


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


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution, background_heightmap=None, median_filter_pixels=5, color_median_filter_pixels=5):
    """ Note:
    Arg median_filter_pixels is used for the depth image. 
    Arg color_median_filter_pixels is used for the color image.
    """
    
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
        min_z = np.nanmin(depth_heightmap)
        if min_z < 0:
            depth_heightmap = np.clip(depth_heightmap, 0, None)
            if min_z < -0.005:
                print('WARNING: get_heightmap() depth_heightmap contains negative heights with min ' + str(min_z) + ', '
                    'saved depth heightmap png files may be invalid! '
                    'See README.md for instructions to collect the depth heightmap again. '
                    'Clipping the minimum to 0 for now.')

    # Create orthographic top-down-view RGB-D color heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_r[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[0]]
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[1]]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[2]]
    if color_median_filter_pixels > 0:
        color_heightmap_r = ndimage.median_filter(color_heightmap_r, size=color_median_filter_pixels)
        color_heightmap_b = ndimage.median_filter(color_heightmap_b, size=color_median_filter_pixels)
        color_heightmap_g = ndimage.median_filter(color_heightmap_g, size=color_median_filter_pixels)
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)


    return color_heightmap, depth_heightmap

def common_sense_action_failure_heuristic(heightmap, heightmap_resolution=0.002, gripper_width=0.06, min_contact_height=0.02, push_length=0.0, z_buffer=0.01):
    """ Get heuristic scores for the grasp Q value at various pixels. 0 means our model confidently indicates no progress will be made, 1 means progress may be possible.
    """
    pixels_to_dilate = int(np.ceil((gripper_width + push_length)/heightmap_resolution))
    kernel = np.ones((pixels_to_dilate, pixels_to_dilate), np.uint8)
    object_pixels = (heightmap > min_contact_height).astype(np.uint8)
    contactable_regions = cv2.dilate(object_pixels, kernel, iterations=1)

    if push_length > 0.0:
        # For push, skip regions where the gripper would be too high
        regional_maximums = ndimage.maximum_filter(heightmap, (pixels_to_dilate, pixels_to_dilate))
        block_pixels = (heightmap > (regional_maximums - z_buffer)).astype(np.uint8)
        # set all the pixels where the push would be too high to zero,
        # meaning it is not an action which would contact any object
        # the blocks and the gripper width around them are set to zero.
        gripper_width_pixels_to_dilate = int(np.ceil((gripper_width)/heightmap_resolution))
        kernel = np.ones((gripper_width_pixels_to_dilate, gripper_width_pixels_to_dilate), np.uint8)
        push_too_high_pixels = cv2.dilate(block_pixels, kernel, iterations=1)
        contactable_regions[np.nonzero(push_too_high_pixels)] = 0

    return contactable_regions

def common_sense_action_space_mask(depth_heightmap, push_predictions=None, grasp_predictions=None, place_predictions=None, place_dilation=None, show_heightmap=False, color_heightmap=None):
    """ Convert predictions to a masked array indicating if tasks may make progress in this region, based on depth_heightmap.

    The masked arrays will indicate 0 where progress may be possible (no mask applied), and 1 where our model confidently indicates no progress will be made.
    Note the mask values here are the opposite of the common_sense_failure_heuristic() function, so where that function has a mask value of 0, this function has a value of 1. 
    In other words the mask values returned here are equivalent to 1-common_sense_action_failure_heuristic(). 
    This is because in the numpy MaksedArray a True value inticates the data at the corresponding location is INVALID.

    # Returns

    Numpy MaskedArrays push_predictions, grasp_predictions, place_predictions
    """
    # TODO(ahundt) "common sense" dynamic action space parameters should be accessible from the command line
    # "common sense" dynamic action space, mask pixels we know cannot lead to progress
    if push_predictions is not None:
        push_contactable_regions = common_sense_action_failure_heuristic(depth_heightmap, gripper_width=0.04, push_length=0.1)
        # "1 - push_contactable_regions" switches the values to mark masked regions we should not visit with the value 1
        push_predictions = np.ma.masked_array(push_predictions, np.broadcast_to(1 - push_contactable_regions, push_predictions.shape, subok=True))
    if grasp_predictions is not None:
        grasp_contact_regions = common_sense_action_failure_heuristic(depth_heightmap, gripper_width=0.00)
        grasp_predictions = np.ma.masked_array(grasp_predictions, np.broadcast_to(1 - grasp_contact_regions, push_predictions.shape, subok=True))
    if place_predictions is not None:
        place_contact_regions = common_sense_action_failure_heuristic(depth_heightmap, gripper_width=place_dilation)
        place_predictions = np.ma.masked_array(place_predictions, np.broadcast_to(1 - place_contact_regions, push_predictions.shape, subok=True))
    if show_heightmap:
        # visualize the common sense function results
        # show the heightmap
        f = plt.figure()
        # f.suptitle(str(trainer.iteration))
        f.add_subplot(1,4, 1)
        if grasp_predictions is not None:
            plt.imshow(grasp_contact_regions)
        f.add_subplot(1,4, 2)
        if push_predictions is not None:
            plt.imshow(push_contactable_regions)
        f.add_subplot(1,4, 3)
        plt.imshow(depth_heightmap)
        f.add_subplot(1,4, 4)
        if color_heightmap is not None:
            plt.imshow(color_heightmap)
        plt.show(block=True)
    return push_predictions, grasp_predictions, place_predictions


def process_prediction_language_masking(language_data, predictions, show_heightmap=True, color_heightmap=None):
    """
    Adds a language mask to the predictions array.

    language_data: an array with shape [1, 256, 2, 1] which will be processed into a mask
    predictions: masked array or ndarray with prediction values for a specific action
    """

    # Convert inputs to np.ma.masked_array objects if they are inputted as np.ndarrays
    if not np.ma.is_masked(predictions):
        if isinstance(predictions, np.ndarray):
            predictions = np.ma.masked_array(predictions, mask=False)
        else:
            raise TypeError("predictions passed into the process_prediction_language_masking function should be np.ma.masked_array or np.ndarray objects.")
    
    # Extract current masks
    currMask = np.ma.getmask(predictions)

    # Peform data processing on the language model output to convert float values to logits
    # NOTE(zhe) should the function be more generic and take in a reformatted list?
    # language_data should have shape torch([1, 256, 2, 1])
    languageMask = softmax(language_data[0,:,:,0], axis=1)
    languageMask = np.float32(languageMask[:,1] > 0.5).reshape((16,-1))      # using mask index 1 here to use the negative mask.

    # TODO(zhe) Should we erode/dilate the mask array? The current mask lets the whole block pass. We may want to increase or decrease the mask area.

    # Scale language masks to match the prediction array sizes
    languageMask = cv2.resize(languageMask, currMask.shape[1:3], interpolation=cv2.INTER_NEAREST)
    languageMask = np.broadcast_to(languageMask, predictions.shape, subok=True)

    # Catching errors
    assert languageMask.shape == currMask.shape and languageMask.shape == predictions.shape, print("ERROR: Shape missmatch in language masking")

    # Combine language mask with existing masks if necessary
    if currMask is np.ma.nomask:
        predictions.mask = languageMask
    else:
        predictions.mask = 1 - np.logical_and(1 - currMask,  1 - languageMask)

    
    if show_heightmap:
        # visualize the common sense function results
        # show the heightmap
        f = plt.figure()
        # f.suptitle(str(trainer.iteration))
        f.add_subplot(1,2, 1)
        #if predictions is not None:
        plt.imshow(predictions.mask[0,:,:])
        f.add_subplot(1,2, 2)
        if color_heightmap is not None:
            plt.imshow(color_heightmap)
        # f.add_subplot(1,4, 2)
        # if push_predictions is not None:
        #     plt.imshow(push_contactable_regions)
        # f.add_subplot(1,4, 3)
        # plt.imshow(depth_heightmap)
        # f.add_subplot(1,4, 4)
        # if color_heightmap is not None:
        #     plt.imshow(color_heightmap)
        plt.show(block=True)

    return predictions



# TODO(zhe) implement language model masking using language model output. The inputs should already be np.masked_arrays
def common_sense_language_model_mask(language_output, push_predictions=None, grasp_predictions=None, place_predictions=None, color_heightmap=None):
    """ 
    Processes the language output into a mask and combine it with existing masks in prediction arrays
    """

    # language masks are currently for grasp and place only. The push predictions will not be operated upon.
    push_predictions = push_predictions
    grasp_predictions = process_prediction_language_masking(language_output['prev_position'], grasp_predictions, color_heightmap=color_heightmap)
    place_predictions = process_prediction_language_masking(language_output['next_position'], place_predictions, color_heightmap=color_heightmap)


    return push_predictions, grasp_predictions, place_predictions

# Loads a transformer model from a config file
def load_language_model_from_config(configYamlPath, weightsPath):

    # Load config yaml file if possible
    if os.path.exists(configYamlPath):
        with open(configYamlPath) as file:
            config=yaml.load(file, Loader=yaml.FullLoader)
    else:
        raise FileNotFoundError(f'unable to find {configYamlPath}')
    
    # Move model to available gpu
    device = "cpu"
    if config["cuda"] is not None and config["cuda"] >= 0:
        free_gpu_id = get_free_gpu()
        if free_gpu_id > -1:
            device = f"cuda:{free_gpu_id}"

    device = torch.device(device)  
    print(f"Language Model on device {device}") 
    test = torch.ones((1))
    test = test.to(device) 

    # Read the vocab from a json file.
    checkpoint_dir = pathlib.Path(config["checkpoint_dir"])
    print(f"Reading vocab from {checkpoint_dir}")
    if os.path.exists(checkpoint_dir.joinpath('vocab.json')):
        with open(checkpoint_dir.joinpath("vocab.json")) as f1:
            train_vocab = json.load(f1)
    else:
        raise FileNotFoundError(f'unable to find {checkpoint_dir.joinpath("vocab.json")}')
    
    # Load the embedder (type specified in the config.yaml)
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)
    if config['embedder'] == "random":
        embedder = RandomEmbedder(tokenizer, train_vocab, config["embedding_dim"], trainable=True)
    elif config['embedder'] == "glove":
        embedder = GloveEmbedder(tokenizer, train_vocab, config["embedding_file"], config["embedding_dim"], trainable=True) 
    elif config['embedder'].startswith("bert"): 
        embedder = BERTEmbedder(model_name = config["embedder"],  max_seq_len = config["max_seq_length"]) 
    else:
        raise NotImplementedError(f'No embedder {config["embedder"]}') 

    # Initiate the encoder
    encoder = TransformerEncoder(image_size = config["resolution"],
                                 patch_size = config["patch_size"], 
                                 language_embedder = embedder, 
                                 n_layers_shared = config["n_shared_layers"],
                                 n_layers_split  = config["n_split_layers"],
                                 n_classes = 2,
                                 channels = config["channels"], 
                                 n_heads = config["n_heads"],
                                 hidden_dim = config["hidden_dim"],
                                 ff_dim = config["ff_dim"],
                                 dropout = config["dropout"],
                                 embed_dropout = config["embed_dropout"],
                                 output_type = config["output_type"], 
                                 positional_encoding_type = config["pos_encoding_type"],
                                 # device = device,
                                 log_weights = config["test"])

    # Load weights
    print(f'loading model weights from {config["checkpoint_dir"]}') 
    state_dict = torch.load(pathlib.Path(config["checkpoint_dir"]).joinpath("best.th"), map_location = device)
    encoder.load_state_dict(state_dict, strict=True)

    return encoder

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

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

# killeen: this is defining the goal
class StackSequence(object):
    def __init__(self, num_obj, is_goal_conditioned_task=True, trial=0, total_steps=1, color_names=None):
        """ Oracle to choose a sequence of specific color objects to interact with.

        Generates one hot encodings for a list of objects of the specified length.
        Can be used for stacking or simply grasping specific objects.

        # Member Variables

        num_obj: the number of objects to manage. Each object is assumed to be in a list indexed from 0 to num_obj.
        is_goal_conditioned_task: do we care about which specific object we are using
        object_color_sequence: to get the full order of the current stack goal.

        """
        self.num_obj = num_obj
        self.is_goal_conditioned_task = is_goal_conditioned_task
        self.trial = trial
        self.reset_sequence()
        self.total_steps = total_steps
        # TODO(zhe) add list of color names, as an optional argument
        self.color_names = color_names
        self.color_len = len(color_names) if color_names is not None else 0

    def reset_sequence(self):
        """ Generate a new sequence of specific objects to interact with.
        """
        if self.is_goal_conditioned_task:
            # 3 is currently the red block
            # object_color_index = 3
            self.object_color_index = 0

            # Choose a random sequence to stack
            self.object_color_sequence = np.random.permutation(self.num_obj)
            # TODO(ahundt) This might eventually need to be the size of robot.stored_action_labels, but making it color-only for now.
            self.object_color_one_hot_encodings = []
            for color in self.object_color_sequence:
                object_color_one_hot_encoding = np.zeros((self.num_obj))
                object_color_one_hot_encoding[color] = 1.0
                self.object_color_one_hot_encodings.append(object_color_one_hot_encoding)
        else:
            self.object_color_index = None
            self.object_color_one_hot_encodings = None
            self.object_color_sequence = None
        self.trial += 1

    def generate_color_command_string(self):
        """ Generates an English command sentence for stacking the last block in the current stacking sequence
            on top of the second to last block using block colors.

            The command could follow the format:
            Place a {color of object_color_index} on top of {color of (object_color_index - 1)}.
        """
        if self.is_goal_conditioned_task and self.color_names is not None:   # generating commands sentences only work for 
            if self.object_color_index == 0:    # If we are just starting a stack
                firstBlockColor = self.color_names[(self.object_color_index) % self.color_len]
                return f'Start with a {firstBlockColor} block.'
            elif self.object_color_index > 0:    # If we are in the middle of stacking
                colorStrs = {'bottom': self.color_names[self.object_color_sequence[self.object_color_index-1] % self.color_len],
                             'top': self.color_names[self.object_color_sequence[self.object_color_index] % self.color_len]}
                command = f'Place a {colorStrs["top"]} block on top of the highest {colorStrs["bottom"]} block.'
                return command
            else:
                return None
        else:
            return None

    def current_one_hot(self):
        """ Return the one hot encoding for the current specific object.
        """
        return self.object_color_one_hot_encodings[self.object_color_index]

    def sequence_one_hot(self):
        """ Return the one hot encoding for the entire stack sequence.
        """
        return np.concatenate(self.object_color_one_hot_encodings)

    def current_sequence_progress(self):
        """ How much of the current stacking sequence we have completed.

        For example, if the sequence should be [0, 1, 3, 2].
        At initialization this will return [0].
        After one next() calls it will return [0, 1].
        After two next() calls it will return [0, 1, 3].
        After three next() calls it will return [0, 1, 3, 2].
        After four next() calls a new sequence will be generated and it will return one element again.
        """
        if self.is_goal_conditioned_task:
            return self.object_color_sequence[:self.object_color_index+1]
        else:
            return None

    def next(self): 
        self.total_steps += 1
        if self.is_goal_conditioned_task:
            self.object_color_index += 1
            if not self.object_color_index < self.num_obj:
                self.reset_sequence()


def check_row_success(depth_heightmap, block_height_threshold=0.02, row_boundary_length=75, row_boundary_width=18, block_pixel_size=550, prev_z_height=None):
    """ Return if the current arrangement of blocks in the heightmap is a valid row 
    """
    heightmap_trans = np.copy(depth_heightmap)
    heightmap_trans = np.transpose(heightmap_trans)

    heightmaps = (depth_heightmap, heightmap_trans)
    counts = []

    for heightmap in heightmaps:
        # threshold pixels which contain a block
        block_pixels = heightmap > block_height_threshold

        # get positions of all those pixels  
        coords = np.nonzero(block_pixels)
        x = coords[1]
        y = coords[0]
        if x.size == 0 or y.size == 0:
            return False, 0

        # get best fit line y=mx+b
        m, b = np.polyfit(x, y, 1)

        # pick 2 random points on the line and find the unit vector
        x1 = 0
        y1 = int(m*x1 + b)
        x2 = 224
        y2 = int(m*x2 + b)

        l = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        x_unit = (x2-x1)/l
        y_unit = (y2-y1)/l

        # centroid of block_pixels
        centroid = (int(np.mean(x)), int(np.mean(y)))
        
        # get row_boundary_rectangle points
        x1_r = int(centroid[0] - x_unit * row_boundary_length - y_unit * row_boundary_width)
        y1_r = int(centroid[1] - y_unit * row_boundary_length + x_unit * row_boundary_width)
        x2_r = int(centroid[0] + x_unit * row_boundary_length - y_unit * row_boundary_width)
        y2_r = int(centroid[1] + y_unit * row_boundary_length + x_unit * row_boundary_width)
        x3_r = int(centroid[0] + x_unit * row_boundary_length + y_unit * row_boundary_width)
        y3_r = int(centroid[1] + y_unit * row_boundary_length - x_unit * row_boundary_width)
        x4_r = int(centroid[0] - x_unit * row_boundary_length + y_unit * row_boundary_width)
        y4_r = int(centroid[1] - y_unit * row_boundary_length - x_unit * row_boundary_width)

        # create row_boundary_mask
        mask = np.zeros((224,224))
        pts = np.array([[x1_r,y1_r],[x2_r,y2_r],[x3_r,y3_r],[x4_r,y4_r]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(mask, [pts], (255,255,255))
        mask = mask > 0  # convert to bool

        # get all block_pixels inside of row_boundary_rectangle and count them 
        block_pixels_in_row = np.logical_and(mask, block_pixels)
        count = np.count_nonzero(block_pixels_in_row)

        counts.append(count)

    true_count = max(counts[0], counts[1])
    row_size = true_count / block_pixel_size

    if prev_z_height is not None:
        success = row_size > prev_z_height
    else:
        success = True

    print("ROW CHECK PIXEL COUNT: ", true_count, ", success: ", success, ", row size: ", row_size)

    return success, row_size

# function to visualize prediction signal on heightmap (with rotations)
def get_prediction_vis(predictions, heightmap, best_pix_ind, blend_ratio=0.5, \
        prob_exp=1, specific_rotation=None, num_rotations=None):

    best_rot_ind = best_pix_ind[0]
    best_action_xy = best_pix_ind[1:]
    canvas = None

    # clip values <0 or >1
    predictions = np.clip(predictions, 0, 1)

    # apply exponential
    predictions = predictions ** prob_exp

    if specific_rotation is None:
        num_rotations = predictions.shape[0]

        # populate canvas
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()

                # reshape to 224x224 (or whatever image size is), and color
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)

                # if this is the correct rotation, draw circle on action coord
                if rotate_idx == best_rot_ind:
                    # need to flip best_action_xy row and col since cv2.circle reads this as (x, y)
                    prediction_vis = cv2.circle(prediction_vis, (int(best_action_xy[1]),
                        int(best_action_xy[0])), 7, (221,211,238), 2)

                # rotate probability map and image to gripper rotation
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations),
                        reshape=False, order=0).astype(np.uint8)
                background_image = ndimage.rotate(heightmap, rotate_idx*(360.0/num_rotations),
                        reshape=False, order=0).astype(np.uint8)

                # blend image and colorized probability heatmap
                prediction_vis = cv2.addWeighted(cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR),
                        blend_ratio, prediction_vis, 1-blend_ratio, 0)

                # add image to row canvas
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)

            # add row canvas to overall image canvas
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas

    else:
        if num_rotations is None:
            raise ValueError("Must specify number of rotations if providing a specific rotation")

        # reshape to 224x224 (or whatever image size is), and color
        prediction_vis = cv2.applyColorMap((predictions*255).astype(np.uint8), cv2.COLORMAP_JET)

        # need to flip best_pix_ind row and col since cv2.circle reads in as (x, y)
        prediction_vis = cv2.circle(prediction_vis, (int(best_action_xy[1]), int(best_action_xy[0])),
                7, (221,211,238), 2)

        # rotate probability map and image to gripper rotation
        prediction_vis = ndimage.rotate(prediction_vis, best_rot_ind*(360.0/num_rotations),
                reshape=False, order=0).astype(np.uint8)
        background_image = ndimage.rotate(heightmap, best_rot_ind*(360.0/num_rotations),
                reshape=False, order=0).astype(np.uint8)

        # blend image and colorized probability heatmap
        prediction_vis = cv2.addWeighted(cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR),
                blend_ratio, prediction_vis, 1-blend_ratio, 0)

        return prediction_vis

def compute_demo_dist(preds, example_actions, metric='l2'):
    """
    Function to evaluate l2 distance and generate demo-signal mask
    """

    # TODO(adit98) see if we should use cos_sim instead of l2_distance as low-level distance metric
    def cos_sim(test_feat, demo_action_embed):
        """
        Helper function to compute cosine similarity.
        Arguments:
            test_feat: pixel-wise embeddings output by NN for test env
            demo_action_embed: embedding of demo action
        """
        # no need to normalize since we are only concerned with relative values
        cos_sim = np.sum(np.multiply(test_feat, demo_action_embed), axis=1)
        norm_factor = np.linalg.norm(test_feat, axis=1) + 1e-4
        return (cos_sim / norm_factor)

    # reshape each example_action to 1 x 64 x 1 x 1
    for i in range(len(example_actions)):
        # check if policy was supplied (entry will be None if it wasn't)
        if example_actions[i][0] is None:
            continue

        # get actions and expand dims
        actions = np.expand_dims(np.stack(example_actions[i]), (1, 3, 4))

        # reshape and update list
        example_actions[i] = actions

    # get mask for each model
    masks = []
    mask_shape = None
    exit = True
    for pred in preds:
        if pred is not None:
            mask = (pred == np.zeros([1, 64, 1, 1])).all(axis=1)
            mask_shape = mask.shape
            masks.append(mask)
            exit = False

        else:
            masks.append(None)

    # exit if no policies were provided
    if exit:
        raise ValueError("Must provide at least one model")

    # calculate distance between example action embedding and preds for each policy and demo
    dists = []
    for ind, actions in enumerate(example_actions):
        for i, action in enumerate(actions):
            # if policy not supplied, insert pixel-wise array of inf distance
            if metric == 'l2':
                if action is None:
                    dists.append(np.ones(mask_shape) * np.inf)
                    continue

                # calculate pixel-wise l2 distance (16x224x224)
                dist = np.sum(np.square(action - preds[ind]), axis=1)
                invert = True

            elif metric == 'cos_sim':
                if action is None:
                    dists.append(np.ones(mask_shape) * np.NINF)
                    continue

                dist = cos_sim(preds[ind], action)
                #print('Policy Number:', ind, '| Primitive Action:', i, '| Best Match Ind:',
                #        np.unravel_index(np.argmax(dist), dist.shape), '| Similarity:', np.max(dist))
                invert = False

            # TODO(adit98) UMAP distance?
            else:
                raise NotImplementedError

            if invert:
                # set all masked spaces to have max l2 distance (select appropriate mask from list of masks)
                dist[masks[ind]] = np.max(dist) * 1.1

            else:
                # set all masked spaces to have min similarity (select appropriate mask from list of masks)
                dist[masks[ind]] = np.min(dist) * 0.9

            #print('Post-Mask | Policy Number:', ind, '| Primitive Action:', i, '| Best Match Ind:',
            #        np.unravel_index(np.argmax(dist), dist.shape), '| Similarity:', np.max(dist))

            # append to dists list
            dists.append(dist)

    # stack pixel-wise distance array per policy (4x16x224x224)
    dists = np.stack(dists)

    if invert:
        # find overall minimum distance across all policies and get index
        match_ind = np.unravel_index(np.argmin(dists), dists.shape)
    else:
        # find overall maximum similarity across all policies and get index
        match_ind = np.unravel_index(np.argmax(dists), dists.shape)

    #print("Selected match_ind:", match_ind)

    # select distance array for policy which contained minimum distance index
    dist = dists[match_ind[0]]

    # discard first dimension of match_ind to get it in the form (theta, y, x)
    match_ind = match_ind[1:]

    # make dist >=0 and max_normalize
    dist = dist - np.min(dist)
    dist = dist / np.max(dist)

    # if our distance metric returns high values for values that are far apart, we need to invert (for viz)
    if invert:
        # invert values of dist so that large values indicate correspondence
        im_mask = 1 - dist

    else:
        im_mask = dist

    return im_mask, match_ind

# TODO(adit98) implement this
def compute_cc_dist(test_preds, demo_preds):
    """
    Function to evaluate l2 distance and generate demo-signal mask
    """

    # TODO(adit98) see if we should use cos_sim instead of l2_distance as low-level distance metric
    def cos_sim(pix_preds, best_pred):
        """
        Helper function to compute cosine similarity.
        Arguments:
            pix_preds: pixel-wise embedding array
            best_pred: template embedding vector
        """
        best_pred = np.expand_dims(best_pred, (0, 2, 3))
        cos_sim = np.multiply(pix_preds, best_pred)
        return cos_sim

    # reshape each example_action to 1 x 64 x 1 x 1
    for i in range(len(example_actions)):
        # check if policy was supplied (entry will be None if it wasn't)
        if example_actions[i][0] is None:
            continue

        # get actions and expand dims
        actions = np.expand_dims(np.stack(example_actions[i]), (1, 3, 4))

        # reshape and update list
        example_actions[i] = actions

    # get mask from first available model (NOTE(adit98) see if we need a different strategy for this)
    masks = []
    mask_shape = None
    exit = True
    for pred in preds:
        if pred is not None:
            mask = (pred == np.zeros([1, 64, 1, 1])).all(axis=1)
            mask_shape = mask.shape
            masks.append(mask)
            exit = False

        else:
            masks.append(None)

    # exit if no policies were provided
    if exit:
        raise ValueError("Must provide at least one model")

    # calculate l2 distance between example action embedding and preds for each policy and demo
    l2_dists = []
    for ind, actions in enumerate(example_actions):
        # check if policy was supplied (entry will be [None, None] if it wasn't)
        if actions[0] is None:
            # if policy not supplied, insert pixel-wise array of inf distance
            # apply 
            l2_dists.append(np.ones(mask_shape) * np.inf)
            continue

        for action in actions:
            # calculate pixel-wise l2 distance (16x224x224)
            dist = np.sum(np.square(action - preds[ind]), axis=1)

            # set all masked spaces to have max l2 distance
            # select appropriate mask from list of masks
            dist[masks[ind]] = np.max(dist) * 1.1

            # append to l2_dists list
            l2_dists.append(dist)

    # stack pixel-wise distance array per policy (4x16x224x224)
    l2_dists = np.stack(l2_dists)

    # find overall minimum distance across all policies and get index
    match_ind = np.unravel_index(np.argmin(l2_dists), l2_dists.shape)

    # select distance array for policy which contained minimum distance index
    l2_dist = l2_dists[match_ind[0]]

    # discard first dimension of match_ind to get it in the form (theta, y, x)
    match_ind = match_ind[1:]

    # make l2_dist >=0 and max_normalize
    l2_dist = l2_dist - np.min(l2_dist)
    l2_dist = l2_dist / np.max(l2_dist)

    # invert values of l2_dist so that large values indicate correspondence
    im_mask = 1 - l2_dist

    return im_mask, match_ind
