import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter

from data_io.env import load_env_config, load_path, load_env_img
from data_io.env import load_template
from env_config.definitions.landmarks import get_landmark_name_to_index, NUM_LANDMARKS, get_null_landmark_name
from env_config.definitions.nlp_templates import N_LANDMARKS, N_SIDES, get_side_name2idx
from data_io.instructions import split_instruction, clean_instruction, words_to_terms, load_landmark_alignments, get_instruction_segment

import learning.datasets.top_down_dataset as tdd
from learning.datasets.masking import get_obs_mask_every_n_and_segstart, get_obs_mask_segstart
from learning.datasets.dynamic_ground_truth import get_dynamic_ground_truth_v2
from learning.inputs.vision import standardize_image, standardize_2d_prob_dist
from learning.models.semantic_map.pinhole_camera_inv import PinholeCameraProjection
from data_io.units import UnrealUnits
from transformations import cf_to_img, pos_m_to_px
from visualization import Presenter
from learning.inputs.pose import Pose, get_noisy_poses_np, stack_poses_np

import parameters.parameter_server as P

from utils.simple_profiler import SimpleProfiler


"""
This file contains auxiliary data providers.
When rolling out the oracle in the environment, we collect a dataset of trajectories of samples.
Each sample is an image, pose and instruction, along with metadata about which environment and instruction
segment the sample came from.

Given this metadata, the functions in this file are used to collect and return various metadata about
the 
"""

PROVIDER_LM_POS_DATA = "lm_pos_lm_indices_fpv"
PROVIDER_GOAL_POS = "goal_loc"
PROVIDER_TRAJECTORY_GROUND_TRUTH_STATIC = "trajectory_map_static"
PROVIDER_TRAJECTORY_GROUND_TRUTH_DYNAMIC = "trajectory_map_dynamic"
PROVIDER_TRAJECTORY_GROUND_TRUTH_DYNAMIC_NOISY = "trajectory_map_dynamic_noisy"
PROVIDER_LANG_TEMPLATE = "mentioned_tplt"
PROVIDER_TOP_DOWN_IMAGES = "top_down_images"
PROVIDER_ROT_TOP_DOWN = "rot_top_down"
PROVIDER_LANDMARKS_MENTIONED = "lm_mentioned"
PROVIDER_TRAJ_HISTORY = "past_trajectory_map"
PROVIDER_NOISY_POSES = "noisy_poses"
PROVIDER_START_POSES = "start_poses"
PROVIDER_POSE_NOISE = "pose_noise"

LANDMARKS_ON_FLOOR = True

ADD_NULL_LANDMARK = True


def draw_null_landmark_pos(landmark_positions):
    """
    Given an array of real landmark positions, finds a place far enough away from all
    other landmarks
    :param landmark_positions:
    :return:
    """
    world_size = P.get_current_parameters()["Setup"]["world_size_m"]
    dst_t = world_size * 0.2
    pos_good = False
    while not pos_good:
        new_pos = np.random.uniform(0, world_size, 2)
        pos_good = min([np.linalg.norm(new_pos - p[:2]) for p in landmark_positions]) > dst_t
    return np.asarray([new_pos[0], new_pos[1], 0])


def get_landmark_locations_airsim(config_json):
    landmark_names = []
    landmark_positions = []
    units = UnrealUnits()
    for i, landmarkName in enumerate(config_json["landmarkName"]):
        x_pos = config_json["xPos"][i]
        y_pos = config_json["zPos"][i]
        pt = np.asarray([x_pos, y_pos])
        pt_as = np.zeros(3)
        pt_as[0:2] = units.pos2d_to_as(pt)
        # TODO: Grab this from the parameter server
        pt_as[2] = 0.0 if LANDMARKS_ON_FLOOR else -1.0  # Landmarks assumed to be floating 1m above ground.
        landmark_names.append(landmarkName)
        landmark_positions.append(pt_as)

    if ADD_NULL_LANDMARK:
        null_pos = draw_null_landmark_pos(landmark_positions)
        landmark_names.append(get_null_landmark_name())
        landmark_positions.append(null_pos)

    name2idx = get_landmark_name_to_index(add_empty=ADD_NULL_LANDMARK)
    landmark_indices = [name2idx[name] for name in landmark_names]

    return landmark_names, landmark_indices, landmark_positions


def get_mentioned_landmarks_nl(str_instruction):
    thesaurus = load_landmark_alignments()
    if thesaurus is None:
        return [], []
    split_instr = split_instruction(clean_instruction(str_instruction))
    word2term = thesaurus["word2term"]
    term_groundings = thesaurus["term_groundings"]
    lm_name2index = get_landmark_name_to_index()

    # Map each word in the instruction to it's corresponding term:
    split_instr_terms = words_to_terms(split_instr, word2term)

    mentioned_landmark_names = set()

    # For each term, find all the landmarks that have been mentioned
    for term in split_instr_terms:
        for landmark_name in term_groundings[term]["landmarks"]:
            mentioned_landmark_names.add(landmark_name)

    mentioned_landmark_names = list(mentioned_landmark_names)
    mentioned_landmark_indices = [lm_name2index[name] for name in mentioned_landmark_names]
    return mentioned_landmark_names, mentioned_landmark_indices


def any_string_is_substring(stringlist, str):
    appears = False
    for referent in stringlist:
        if str.find(referent) > 0:
            appears = True
    return appears


def get_mentioned_landmarks_tplt(str_instruction):
    mentioned_names = set()
    for landmark_name, referents in N_LANDMARKS.items():
        if any_string_is_substring(referents, str_instruction):
            mentioned_names.add(landmark_name)
    mentioned_names = list(mentioned_names)
    lm_name2index = get_landmark_name_to_index()
    mentioned_indices = [lm_name2index[name] for name in mentioned_names]
    return mentioned_names, mentioned_indices


def get_mentioned_landmark_side_tplt(env_id):
    template = load_template(env_id)
    mentioned_lm = template["landmark1"]
    lm_name2index = get_landmark_name_to_index()
    mentioned_index = lm_name2index[mentioned_lm]

    mentioned_side = template["side"]
    side_name2index = get_side_name2idx()
    side_idx = side_name2index[mentioned_side]

    return mentioned_lm, mentioned_index, mentioned_side, side_idx


def get_mentioned_sides_tplt(str_instruction):

    for i, side_name in enumerate(sorted(N_SIDES.keys())):
        referents = N_SIDES[side_name]
        if any_string_is_substring(referents, str_instruction):
            return side_name, i
    return 0, 0


def get_top_down_image_env(env_id, map_w, map_h, img_w, img_h):
    """
    To be called externally to retrieve a top-down environment image oriented with the start of the requested segment
    :param env_id:  environment id
    :return:
    """
    env_image_in = load_env_img(env_id, map_w, map_h)
    # If we need to return a bigger image resolution than we loaded
    if map_w != img_w or map_h != img_h:
        env_image = np.zeros([img_h, img_w, env_image_in.shape[2]])
        env_image[0:map_h, 0:map_w, :] = env_image_in
    else:
        env_image = env_image_in
    #path_img = cf_to_img(path, [env_image.shape[0], env_image.shape[1]])
    #self.plot_path_on_img(env_image, path_img)
    env_image = standardize_image(env_image)
    env_img_t = torch.from_numpy(env_image).unsqueeze(0).float()
    #presenter = Presenter()
    #presenter.show_image(env_img_t[0], "data_img", torch=True, scale=1)
    return env_img_t


def get_top_down_ground_truth_static_ego(env_id, start_idx, img_w, img_h, map_w, map_h):
    """
    Returns the ground-truth label oriented in the global map frame
    :param env_id:
    :param start_idx:
    :param img_w:
    :param img_h:
    :param map_w:
    :param map_h:
    :return:
    """
    path = load_path(env_id)
    #instruction_segments = [self.all_instr[env_id][set_idx]["instructions"][seg_idx]]

    start_pt, dir_yaw = tdd.get_start_pt_and_yaw(path, start_idx, map_w, map_h, 0)
    affine = tdd.get_affine_matrix(start_pt, dir_yaw, img_w, img_h)

    seg_labels = np.zeros([img_w, img_h, 2]).astype(float)
    path_in_img = cf_to_img(path, np.array([map_w, map_h]))

    #gauss_sigma = map_w / 96
    gauss_sigma = map_w / 32

    seg_labels[:, :, 0] = tdd.plot_path_on_img(seg_labels[:, :, 0], path_in_img)
    if len(path_in_img) > 1:
        seg_labels[:, :, 1] = tdd.plot_dot_on_img(seg_labels[:, :, 1], path_in_img[-1], gauss_sigma)

    seg_labels_rot = tdd.apply_affine(seg_labels, affine, img_w, img_h)
    seg_labels_rot[:, :, 0] = gaussian_filter(seg_labels_rot[:, :, 0], gauss_sigma)
    seg_labels_rot[:, :, 1] = gaussian_filter(seg_labels_rot[:, :, 1], gauss_sigma)

    DEBUG = True
    if DEBUG:
        cv2.imshow("l_traj", seg_labels_rot[:, :, 0])
        cv2.imshow("l_endpt", seg_labels_rot[:, :, 1])
        cv2.waitKey(0)

    # Standardize both channels separately (each has mean zero, unit variance)
    seg_labels_path = standardize_2d_prob_dist(seg_labels_rot[:, :, 0:1])
    seg_labels_endpt = standardize_2d_prob_dist(seg_labels_rot[:, :, 1:2])

    seg_labels_rot = np.concatenate((seg_labels_path, seg_labels_endpt), axis=0)

    seg_labels_t = torch.from_numpy(seg_labels_rot).unsqueeze(0).float()
    return seg_labels_t


def resolve_and_get_ground_truth_static_global(env_id, set_idx, seg_idx, map_size_px, world_size_px):
    seg = get_instruction_segment(env_id, set_idx, seg_idx)
    start_idx = seg["start_idx"]
    end_idx = seg["end_idx"]
    return get_top_down_ground_truth_static_global(env_id, start_idx, end_idx,
                                                   map_size_px, map_size_px, world_size_px, world_size_px)


def get_top_down_ground_truth_static_global(env_id, start_idx, end_idx, img_w, img_h, map_w, map_h):
    """
    Returns the ground-truth label oriented in the global map frame
    :param env_id:
    :param start_idx:
    :param img_w:
    :param img_h:
    :param map_w:
    :param map_h:
    :return:
    """
    path = load_path(env_id)
    path = path[start_idx:end_idx]
    #instruction_segments = [self.all_instr[env_id][set_idx]["instructions"][seg_idx]]

    seg_labels = np.zeros([img_w, img_h, 2]).astype(float)
    path_in_img = cf_to_img(path, np.array([map_w, map_h]))
    gauss_sigma = map_w / 96

    seg_labels[:, :, 0] = tdd.plot_path_on_img(seg_labels[:, :, 0], path_in_img)
    if len(path_in_img) > 1:
        seg_labels[:, :, 1] = tdd.plot_dot_on_img(seg_labels[:, :, 1], path_in_img[-1], gauss_sigma)

    seg_labels[:, :, 0] = gaussian_filter(seg_labels[:, :, 0], gauss_sigma)
    seg_labels[:, :, 1] = gaussian_filter(seg_labels[:, :, 1], gauss_sigma)

    # Standardize both channels separately (each has mean zero, unit variance)
    seg_labels_path = standardize_2d_prob_dist(seg_labels[:, :, 0:1])
    seg_labels_endpt = standardize_2d_prob_dist(seg_labels[:, :, 1:2])

    DEBUG = False
    if DEBUG:
        cv2.imshow("l_traj", seg_labels_path[0, :, :])
        cv2.imshow("l_endpt", seg_labels_endpt[0, :, :])
        cv2.waitKey(10)

    seg_labels = np.concatenate((seg_labels_path, seg_labels_endpt), axis=0)

    seg_labels_t = torch.from_numpy(seg_labels).unsqueeze(0).float()
    return seg_labels_t


def get_top_down_ground_truth_dynamic_global(env_id, start_idx, end_idx, drone_pos_as, img_w, img_h, map_w, map_h):
    """
    Returns the ground-truth label oriented in the global map frame
    :param env_id:
    :param start_idx:
    :param img_w:
    :param img_h:
    :param map_w:
    :param map_h:
    :return:
    """
    PROFILE = False
    prof = SimpleProfiler(False, PROFILE)
    path = load_path(env_id, anno=True)
    #print(len(path), start_idx, end_idx)

    path = path[start_idx:end_idx]
    #instruction_segments = [self.all_instr[env_id][set_idx]["instructions"][seg_idx]]

    prof.tick("load_path")
    units = UnrealUnits(1.0)
    drone_pos_cf = units.pos3d_from_as(drone_pos_as)

    #print("Dynamic ground truth for ", env_id, start_idx, end_idx)
    gt_dynamic = get_dynamic_ground_truth_v2(path, drone_pos_cf[:2])
    #Presenter().plot_path(env_id, [path[start_idx:end_idx], gt_dynamic])

    prof.tick("gen_gt_path")

    seg_labels = np.zeros([img_w, img_h, 2]).astype(float)
    path_in_img = cf_to_img(gt_dynamic, np.array([map_w, map_h]))
    gauss_sigma = map_w / 96

    seg_labels[:, :, 0] = tdd.plot_path_on_img(seg_labels[:, :, 0], path_in_img)
    if len(path_in_img) > 1:
        seg_labels[:, :, 1] = tdd.plot_dot_on_img(seg_labels[:, :, 1], path_in_img[-1], gauss_sigma)

    prof.tick("plot_path")

    seg_labels[:, :, 0] = gaussian_filter(seg_labels[:, :, 0], gauss_sigma)
    seg_labels[:, :, 1] = gaussian_filter(seg_labels[:, :, 1], gauss_sigma)

    # Standardize both channels separately (each has mean zero, unit variance)
    seg_labels_path = standardize_2d_prob_dist(seg_labels[:, :, 0:1])
    seg_labels_endpt = standardize_2d_prob_dist(seg_labels[:, :, 1:2])

    prof.tick("process_img")

    DEBUG = False
    if DEBUG:
        gt_path_in_img = cf_to_img(path, np.asarray([map_w, map_h]))
        dbg_labels_gt = np.zeros([img_w, img_h, 1])
        dbg_labels_gt[:, :, 0] = tdd.plot_path_on_img(dbg_labels_gt[:, :, 0], gt_path_in_img)
        Presenter().show_image(dbg_labels_gt, "dbg", torch=False, waitkey=10, scale=4)
        Presenter().show_image(torch.from_numpy(seg_labels_path), "l_path", torch=True, waitkey=10, scale=4)
        Presenter().show_image(torch.from_numpy(seg_labels_endpt), "l_endp", torch=True, waitkey=100, scale=4)

    seg_labels = np.concatenate((seg_labels_path, seg_labels_endpt), axis=0)

    seg_labels_t = torch.from_numpy(seg_labels).unsqueeze(0).float()

    prof.tick("prep_out")
    prof.print_stats()

    return seg_labels_t


def __get_goal_location_airsim(goal):
    units = UnrealUnits()
    goal_x = goal[0]
    goal_y = goal[1]
    pt = np.asarray([goal_x, goal_y])
    pt_as = np.zeros(2)
    pt_as[0:2] = units.pos2d_to_as(pt)
    return pt_as


def provider_lm_pos_lm_indices_fpv(segment_data, data):
    """
    Data provider that gives the positions and indices of all landmarks visible in the FPV image.
    :param segment_data: segment dataset for which to provide data
    :return: ("lm_pos", lm_pos) - lm_pos is a list (over timesteps) of lists (over landmarks visible in image) of the
                landmark locations in image pixel coordinates
             ("lm_indices", lm_indices) - lm_indices is a list (over timesteps) of lists (over landmarks visible in image)
                of the landmark indices for every landmark included in lm_pos. These are the landmark classifier labels
    """
    env_id = segment_data[0]["metadata"]["env_id"]
    domain = segment_data[0]["metadata"]["domain"]

    #if INSTRUCTIONS_FROM_FILE:
    #    env_instr = load_instructions(env_id)

    conf_json = load_env_config(env_id)
    all_landmark_indices = get_landmark_name_to_index()
    landmark_names, landmark_indices, landmark_pos = get_landmark_locations_airsim(conf_json)

    params = P.get_current_parameters().get("Model") or P.get_current_parameters().get("ModelPVN").get("Stage1")
    projector = PinholeCameraProjection(
        map_size_px=params["global_map_size"],
        world_size_px=params["world_size_px"],
        world_size_m=params["world_size_m"],
        img_x=params["img_w"],
        img_y=params["img_h"],
        cam_fov=params["cam_h_fov"],
        domain=domain,
        use_depth=False
        )
    traj_len = len(segment_data)

    lm_pos_fpv = []
    lm_indices = []
    lm_mentioned = []
    lm_pos_map = []

    for timestep in range(traj_len):
        t_lm_pos_fpv = []
        t_lm_indices = []
        t_lm_mentioned = []
        t_lm_pos_map = []

        if segment_data[timestep]["state"] is not None:
            cam_pos = segment_data[timestep]["state"].get_cam_pos_3d()
            cam_rot = segment_data[timestep]["state"].get_cam_rot()
            instruction_str = segment_data[timestep]["instruction"]
            mentioned_landmark_names, mentioned_landmark_indices = get_mentioned_landmarks_nl(instruction_str)

            for i, landmark_in_world in enumerate(landmark_pos):
                landmark_idx = landmark_indices[i]
                landmark_in_img, landmark_in_cam, status = projector.world_point_to_image(cam_pos, cam_rot, landmark_in_world)
                this_lm_mentioned = 1 if landmark_idx in mentioned_landmark_indices else 0

                # This is None if the landmark is behind the camera.
                if landmark_in_img is not None:
                    # presenter.save_image(images[timestep], name="tmp.png", torch=True, draw_point=landmark_in_img)
                    t_lm_pos_fpv.append(landmark_in_img[0:2])
                    t_lm_pos_map.append(landmark_in_world[0:2])
                    t_lm_indices.append(landmark_idx)
                    t_lm_mentioned.append(this_lm_mentioned)

        if len(t_lm_pos_fpv) > 0:
            t_lm_pos_fpv = torch.from_numpy(np.asarray(t_lm_pos_fpv)).float()
            t_lm_pos_map = torch.from_numpy(np.asarray(t_lm_pos_map)).float()
            t_lm_indices = torch.from_numpy(np.asarray(t_lm_indices)).long()
            t_lm_mentioned = torch.from_numpy(np.asarray(t_lm_mentioned)).long()
        else:
            t_lm_pos_fpv = None
            t_lm_pos_map = None
            t_lm_indices = None
            t_lm_mentioned = None

        lm_pos_fpv.append(t_lm_pos_fpv)
        lm_pos_map.append(t_lm_pos_map)
        lm_indices.append(t_lm_indices)
        lm_mentioned.append(t_lm_mentioned)

    return [("lm_pos_fpv", lm_pos_fpv), ("lm_indices", lm_indices), ("lm_mentioned", lm_mentioned), ("lm_pos_map", lm_pos_map)]


def provider_goal_pos_map(segment_data, data):
    """
        Data provider that gives the positions and indices of all landmarks visible in the FPV image.
        :param segment_data: segment dataset for which to provide data
        :return: ("lm_pos", lm_pos) - lm_pos is a list (over timesteps) of lists (over landmarks visible in image) of the
                    landmark locations in image pixel coordinates
                 ("lm_indices", lm_indices) - lm_indices is a list (over timesteps) of lists (over landmarks visible in image)
                    of the landmark indices for every landmark included in lm_pos. These are the landmark classifier labels
        """

    env_id = segment_data[0]["metadata"]["env_id"]
    path = load_path(env_id)

    traj_len = len(segment_data)

    goal_loc = []
    for timestep in range(traj_len):
        if segment_data[timestep] is None:
            goal_loc.append(np.asarray([0.0, 0.0]))
            continue

        set_idx = segment_data[timestep]["metadata"]["set_idx"]
        seg_idx = segment_data[timestep]["metadata"]["seg_idx"]

        seg = get_instruction_segment(env_id, set_idx, seg_idx)
        end_idx = seg["end_idx"]

        if end_idx < len(path):
            end_pt = path[end_idx]
        else:
            end_pt = path[-1]
        goal_as = __get_goal_location_airsim(end_pt)
        goal_loc.append(goal_as)

    goal_loc = np.asarray(goal_loc)
    goal_loc_t = torch.from_numpy(goal_loc).float()

    return [("goal_loc", goal_loc_t)]


def provider_mentioned_lang_template(segment_data, data):
    traj_len = len(segment_data)
    all_mentioned_lm_indices = []
    all_mentioned_side_indices = []

    lm_name, lm_idx, side_name, side_idx = get_mentioned_landmark_side_tplt(segment_data[0]["metadata"]["env_id"])

    for timestep in range(traj_len):
        if segment_data[timestep] is not None:
            # TODO: for natural language, we'll use the NL functions above, instead of the tlpt ones
            all_mentioned_lm_indices.append(lm_idx)
            all_mentioned_side_indices.append(side_idx)
        else:
            all_mentioned_lm_indices.append(0)
            all_mentioned_side_indices.append(0)

    amlit = torch.from_numpy(np.asarray(all_mentioned_lm_indices))
    amsit = torch.from_numpy(np.asarray(all_mentioned_side_indices))

    return [("lm_mentioned_tplt", amlit), ("side_mentioned_tplt", amsit)]


def provider_trajectory_ground_truth(segment_data, data, kind="static"):
    # For now, use only the first label
    traj_len = len(segment_data)
    env_id = segment_data[0]["metadata"]["env_id"]
    labels = []

    # TODO: This could be more general than PVN model, but for now it's really not gonna be
    model_params = P.get_current_parameters()["ModelPVN"]["Stage1"]
    plan_every_n_steps = model_params["plan_every_n_steps"]
    #m_size = model_params["local_map_size"]
    m_size = model_params["global_map_size"]
    w_size = model_params["world_size_px"]

    # True for planning timesteps, False for the other timesteps
    obs_mask = get_obs_mask_every_n_and_segstart(plan_every_n_steps, segment_data)
    firstseg_mask = get_obs_mask_segstart(segment_data)

    for timestep in range(traj_len):
        # TODO: Shouldn't do this for every single timestep, otherwise it takes really long!
        if segment_data[timestep] is not None and obs_mask[timestep]:
            md = segment_data[timestep]["metadata"]
            seg = get_instruction_segment(md["env_id"], md["set_idx"], md["seg_idx"])
            start_idx = seg["start_idx"]
            end_idx = seg["end_idx"]

            if kind == "dynamic":
                pos = segment_data[timestep]["state"].state[9:12]
                labels_t = get_top_down_ground_truth_dynamic_global(env_id, start_idx, end_idx, pos, m_size, m_size, w_size, w_size)

            elif kind == "dynamic_noisy":
                assert "noisy_poses" in data, "Noisy poses must be computed before computing dynamic ground truth!"
                pos = data["noisy_poses"][timestep].position
                labels_t = get_top_down_ground_truth_dynamic_global(env_id, start_idx, end_idx, pos, m_size, m_size, w_size, w_size)

            elif kind == "static":
                labels_t = get_top_down_ground_truth_static_global(env_id, start_idx, end_idx, m_size, m_size, w_size, w_size)

            else:
                raise Exception("Unknown trajectory ground truth kind")
            # append CxHxW
            labels.append(labels_t[0])
            # TODO: for natural language, we'll use the NL functions above, instead of the tlpt ones
        #else:
        #    labels.append(labels[-1])

    # create labels SxCxHxW
    labels = torch.stack(labels, dim=0)

    return [("traj_ground_truth", labels), ("plan_mask", obs_mask), ("firstseg_mask", firstseg_mask)]


def provider_trajectory_ground_truth_static(segment_data, data):
    return provider_trajectory_ground_truth(segment_data, data, "static")


def provider_trajectory_ground_truth_dynamic(segment_data, data):
    return provider_trajectory_ground_truth(segment_data, data, "dynamic")


def provider_trajectory_ground_truth_dynamic_noisy(segment_data, data):
    return provider_trajectory_ground_truth(segment_data, data, "dynamic_noisy")


def provider_top_down_images(segment_data, data):
    traj_len = len(segment_data.metadata)
    env_id = segment_data.metadata[0]["env_id"]

    top_down_images = []
    #env_image is CxHxW
    env_image = get_top_down_image_env(env_id, 256, 256, 512, 512)[0]

    prev_seg = {"env_id": -1, "set_idx": -1, "seg_idx": -1}
    for timestep in range(1):
        top_down_images.append(env_image)

    # SxCxHxW
    top_down_images_t = torch.stack(top_down_images, dim=0)

    return [("top_down_images", top_down_images_t)]


def provider_rot_top_down_images(segment_data, data):
    env_id = segment_data.metadata[0]["env_id"]

    path = load_path(env_id)
    env_image = load_env_img(env_id, 256, 256)

    top_down_images = []
    top_down_labels = []

    for md in segment_data.metadata:
        if md is None:
            break
        set_idx = md["set_idx"]
        seg_idx = md["seg_idx"]

        instr_seg = get_instruction_segment(env_id, set_idx, seg_idx)
        start_idx = instr_seg["start_idx"]
        end_idx = instr_seg["end_idx"]

        start_pt, dir_yaw = tdd.get_start_pt_and_yaw(path, start_idx, 256, 256, 0)
        affine = tdd.get_affine_matrix(start_pt, dir_yaw, 512, 512)
        seg_img_t = tdd.gen_top_down_image(env_image, affine, 512, 512, 256, 256)
        seg_labels_t = tdd.gen_top_down_labels(path[start_idx:end_idx], affine, 512, 512, 256, 256, True, True)

        seg_labels_t = F.max_pool2d(Variable(seg_labels_t), 8).data

        top_down_images.append(seg_img_t)
        top_down_labels.append(seg_labels_t)

    tdimg_t = torch.cat(top_down_images, dim=0)
    tdlab_t = torch.cat(top_down_labels, dim=0)

    return[("top_down_images", tdimg_t), ("traj_ground_truth", tdlab_t)]


def provider_landmarks_mentioned(segment_data, data):
    traj_len = len(segment_data)

    mentioned_lm_indices = []
    mentioned_lm_names = []
    mentioned_lm_stack = []

    for timestep in range(traj_len):
        if segment_data[timestep] is not None:
            mentioned_lm_t = torch.zeros([NUM_LANDMARKS]).long()

            instruction_str = segment_data[timestep]["instruction"]
            mentioned_landmark_names, mentioned_landmark_indices = get_mentioned_landmarks_nl(instruction_str)
            mentioned_lm_indices.append(mentioned_landmark_indices)
            mentioned_lm_names.append(mentioned_lm_names)

            # TODO: Why is this a double-list?
            for index in mentioned_lm_indices[0]:
                mentioned_lm_t[index] = 1

            mentioned_lm_stack.append(mentioned_lm_t)

    mentioned_lms_t = torch.stack(mentioned_lm_stack, dim=0)

    return [("lang_lm_mentioned_indices", mentioned_lm_indices),
            ("lang_lm_mentioned_names", mentioned_lm_names),
            ("lang_lm_mentioned", mentioned_lms_t)]


def provider_past_trajectory(segment_data, data):
    traj_len = len(segment_data)

    canvas = np.zeros((64, 64))
    canvases_t = []
    last_pos = None
    for timestep in range(traj_len):
        if segment_data[timestep]["state"] is None:
            break
        pos_as = segment_data.state[timestep].state[9:12]
        pos_map = pos_m_to_px(pos_as[np.newaxis, :], img_size_px=32)[0]
        if last_pos != None:
            coords = [last_pos, pos_map]
            last_pos = pos_map
            tdd.plot_path_on_img(canvas, coords)
            cv2.imshow("past_traje", canvas)
        canvas_t = torch.from_numpy(canvas.copy())
        canvases_t.append(canvas_t)
    canvases_t = torch.stack(canvases_t, dim=0)
    return [("past_trajectory_map", canvases_t)]


def provider_noisy_poses(segment_data, data):
    """
    This provider returns noisy poses of type learning.inputs.Pose
    These noisy poses are used during training to rotate the semantic map by a random angle before predicting visitation
    probabilities as a form of data augmentation.
    :param segment_data:
    :param data:
    :return:
    """
    traj_len = len(segment_data)
    last_pos = None
    clean_poses = []

    model_params = P.get_current_parameters()["ModelPVN"]["Stage1"]
    use_first_pose = model_params["predict_in_start_frame"]

    seg_idx = -1
    first_step = 0
    for timestep in range(traj_len):

        if segment_data[timestep]["state"] is None:
            break
        if segment_data[timestep]["metadata"]["seg_idx"] != seg_idx:
            first_step = timestep
            seg_idx = segment_data[timestep]["metadata"]["seg_idx"]

        if use_first_pose:
            # X["state"] is a DroneState object
            pos_as = segment_data[first_step]["state"].state[9:12]
            rot_as = segment_data[first_step]["state"].state[12:16]
        else:
            pos_as = segment_data[timestep]["state"].state[9:12]
            rot_as = segment_data[timestep]["state"].state[12:16]

        clean_pose = Pose(pos_as, rot_as)
        clean_poses.append(clean_pose)

    params = P.get_current_parameters()["Data"]

    noisy_poses = get_noisy_poses_np(clean_poses, params["noisy_pos_variance"], params["noisy_rot_variance"])
    noisy_poses_t = noisy_poses.to_torch()

    return [("noisy_poses", noisy_poses_t)]


def provider_start_poses(segment_data, data):
    traj_len = len(segment_data)
    start_poses = []

    seg_idx = -2
    for timestep in range(traj_len):
        if segment_data[timestep] is None:
            break
        if segment_data[timestep]["metadata"]["seg_idx"] != seg_idx:
            seg_idx = segment_data[timestep]["metadata"]["seg_idx"]
            pos_as = segment_data[timestep]["state"].state[9:12]
            rot_as = segment_data[timestep]["state"].state[12:16]
            start_pose = Pose(pos_as, rot_as)
        start_poses.append(start_pose)

    start_poses = stack_poses_np(start_poses)
    sart_poses_t = start_poses.to_torch()

    return [("start_poses", sart_poses_t)]


def resolve_data_provider(aux_provider_name):
    """
    Given a name of one of the auxiliary data providers, returns a function that takes a data segment and returns the
    multiple auxiliary data sources
    :param aux_provider_name: one of lm_pos_lm_indices_fpv, lm_pos_lm_indices_map, goal_pos_map, trajectory_map
    :return:
    """
    if aux_provider_name == PROVIDER_LM_POS_DATA:
        return provider_lm_pos_lm_indices_fpv
    elif aux_provider_name == PROVIDER_TRAJECTORY_GROUND_TRUTH_STATIC:
        return provider_trajectory_ground_truth_static
    elif aux_provider_name == PROVIDER_TRAJECTORY_GROUND_TRUTH_DYNAMIC:
        return provider_trajectory_ground_truth_dynamic
    elif aux_provider_name == PROVIDER_TRAJECTORY_GROUND_TRUTH_DYNAMIC_NOISY:
        return provider_trajectory_ground_truth_dynamic_noisy
    elif aux_provider_name == PROVIDER_GOAL_POS:
        return provider_goal_pos_map
    elif aux_provider_name == PROVIDER_LANG_TEMPLATE:
        return provider_mentioned_lang_template
    elif aux_provider_name == PROVIDER_TOP_DOWN_IMAGES:
        return provider_top_down_images
    elif aux_provider_name == PROVIDER_ROT_TOP_DOWN:
        return provider_rot_top_down_images
    elif aux_provider_name == PROVIDER_LANDMARKS_MENTIONED:
        return provider_landmarks_mentioned
    elif aux_provider_name == PROVIDER_TRAJ_HISTORY:
        return provider_past_trajectory
    elif aux_provider_name == PROVIDER_NOISY_POSES:
        return provider_noisy_poses
    elif aux_provider_name == PROVIDER_START_POSES:
        return provider_start_poses


def get_aux_label_names(aux_provider_names):
    """
    :param aux_provider_names:
    :return:
    """
    label_names = []
    for provider in aux_provider_names:
        if provider == PROVIDER_LM_POS_DATA:
            label_names += ["lm_pos_fpv", "lm_pos_map", "lm_indices", "lm_mentioned"]
        elif provider == PROVIDER_GOAL_POS:
            label_names += ["goal_loc"]
        elif provider == PROVIDER_TRAJECTORY_GROUND_TRUTH_STATIC:
            label_names += ["traj_ground_truth", "plan_mask", "firstseg_mask"]
        elif provider == PROVIDER_TRAJECTORY_GROUND_TRUTH_DYNAMIC:
            label_names += ["traj_ground_truth", "plan_mask", "firstseg_mask"]
        elif provider == PROVIDER_TRAJECTORY_GROUND_TRUTH_DYNAMIC_NOISY:
            label_names += ["traj_ground_truth", "plan_mask", "firstseg_mask"]
        elif provider == PROVIDER_LANG_TEMPLATE:
            label_names += ["lm_mentioned_tplt", "side_mentioned_tplt"]
        elif provider == PROVIDER_TOP_DOWN_IMAGES:
            label_names += ["top_down_images"]
        elif provider == PROVIDER_ROT_TOP_DOWN:
            label_names += ["top_down_images", "traj_ground_truth"]
        elif provider == PROVIDER_LANDMARKS_MENTIONED:
            label_names += ["lang_lm_mentioned", "lang_lm_mentioned_indices", "lang_lm_mentioned_names"]
        elif provider == PROVIDER_TRAJ_HISTORY:
            label_names += ["past_trajectory_map"]
        elif provider == PROVIDER_NOISY_POSES:
            label_names += ["noisy_poses"]
        elif provider == PROVIDER_START_POSES:
            label_names += ["start_poses"]

    return label_names


def get_stackable_label_names(aux_provider_names):
    """
    Returns a list of label names that can be stacked as tensors within the collate function.
    Some labels are variable length, some are lists and can't be trivially stacked.
    This should basically include all data that's in form of uniform-length tensors
    :param aux_provider_names:
    :return:
    """
    label_names = []
    for provider in aux_provider_names:
        if provider == PROVIDER_LANG_TEMPLATE:
            label_names += ["lm_mentioned_tplt", "side_mentioned_tplt"]
        elif provider == PROVIDER_TOP_DOWN_IMAGES:
            label_names += ["top_down_images"]
        elif provider == PROVIDER_TRAJECTORY_GROUND_TRUTH_STATIC:
            label_names += ["traj_ground_truth"]
        elif provider == PROVIDER_TRAJECTORY_GROUND_TRUTH_DYNAMIC:
            label_names += ["traj_ground_truth"]
        elif provider == PROVIDER_TRAJECTORY_GROUND_TRUTH_DYNAMIC_NOISY:
            label_names += ["traj_ground_truth"]
        elif provider == PROVIDER_ROT_TOP_DOWN:
            label_names += ["top_down_images", "traj_ground_truth"]
        elif provider == PROVIDER_LANDMARKS_MENTIONED:
            label_names += ["lang_lm_mentioned"]
        elif provider == PROVIDER_TRAJ_HISTORY:
            label_names += ["past_trajectory_map"]
        elif provider == PROVIDER_NOISY_POSES:
            pass
        elif provider == PROVIDER_START_POSES:
            pass

    return label_names