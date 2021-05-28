from collections import namedtuple
from geometry import vec_to_yaw
from transformations import get_affine_rot_2d, get_affine_trans_2d, pos_m_to_px, cf_to_img
from scipy.ndimage.interpolation import affine_transform
import skimage.draw as draw
import skimage.transform as transform

import numpy as np
import random
import scipy as sp
from scipy.ndimage.filters import gaussian_filter
import torch
import cv2
from torch.autograd import Variable
from torch.utils.data import Dataset
from utils.dict_tools import dict_zip, dictlist_append, dict_map

from data_io.instructions import tokenize_instruction, get_all_instructions, get_word_to_token_map, \
    load_landmark_alignments, get_mentioned_landmarks
from data_io.env import load_env_img, load_path, load_env_config, get_landmark_locations_airsim
from learning.inputs.vision import standardize_image, standardize_2d_prob_dist
from learning.inputs.common import empty_float_tensor
from learning.inputs.sequence import sequence_list_to_masked_tensor

from visualization import Presenter

Sample = namedtuple("Sample", ("instruction", "state", "action", "reward", "done", "metadata"))

DEBUG = False


def get_affine_matrix(start_pt, dir_yaw, img_w, img_h):
    img_origin = np.array([img_w / 2, img_h / 2])

    affine_t = get_affine_trans_2d(-start_pt)
    affine_rot = get_affine_rot_2d(-dir_yaw)
    affine_t2 = get_affine_trans_2d(img_origin)

    affine_total = np.dot(affine_t2, np.dot(affine_rot, affine_t))

    return affine_total


def get_start_pt_and_yaw(path, start_idx, map_w, map_h, yaw_rand_range):
    path_img = cf_to_img(path, np.array([map_w, map_h]))
    if start_idx > len(path) - 2:
        return None, None

    start_pt = path_img[start_idx]

    # Due to the way the data is collected, turns result in multiple subsequent points at the same location,
    # which messes up the start orientation. That's why we search for the next point that isn't the same as current pt
    next_idx = start_idx + 1
    next_pt = path_img[next_idx]
    while (next_pt == start_pt).all() and next_idx < len(path) - 1:
        next_idx += 1
        next_pt = path_img[next_idx]

    dir_vec = next_pt - start_pt
    dir_yaw = vec_to_yaw(dir_vec) - np.pi / 2

    if yaw_rand_range > 0:
        dir_yaw_offset = random.gauss(0, yaw_rand_range)
        #dir_yaw_offset = random.uniform(-yaw_rand_range, yaw_rand_range)
        dir_yaw = dir_yaw + dir_yaw_offset
    return start_pt, dir_yaw


def plot_path_on_img(img, path):
    prev_x = None
    prev_y = None
    for coord in path:
        x = int(coord[0])
        y = int(coord[1])
        if prev_x is not None and prev_y is not None:
            rr, cc = draw.line(prev_x, prev_y, x, y)
            rr = np.clip(rr, 0, img.shape[0] - 1)
            cc = np.clip(cc, 0, img.shape[1] - 1)
            img[rr, cc] = 1.0
        prev_x = x
        prev_y = y
    return img


def plot_point_on_img(img, point, size):
    point = [int(point[0]), int(point[1])]
    for i in range(-size, size):
        x = point[0] + i
        if x < 0 or x >= img.shape[0]:
            continue
        for j in range(-size, size):
            y = point[1] + j
            if y < 0 or y >= img.shape[1]:
                continue
            if i**2 + j**2 > size ** 2:
                continue

            img[x][y] = 1.0
    return img


def plot_dot_on_img(img, dot, brightness=1.0):
    dot = [int(dot[0]), int(dot[1])]
    img[dot[0],dot[1]] = brightness
    return img


def swap_affine_xy(affine):
    affine_swap = affine[[1, 0]]
    affine_swap = affine_swap[:, [1,0,2]]
    return affine_swap


def apply_affine(img, affine_mat, crop_w, crop_h):
    # swap x and y axis, because OpenCV uses the y,x addressing convention instead of x,y.
    affine_swap = swap_affine_xy(affine_mat)
    #affine_swap = affine_mat[[0, 1]]
    out_crop_size = np.array([crop_w, crop_h])

    out = cv2.warpAffine(img, affine_swap, tuple(out_crop_size))
    if len(out.shape) < len(img.shape):
        out = np.expand_dims(out, 2)
    return out


def apply_affine_on_pts(pts, affine):
    pts_aff = np.ones((pts.shape[0], 3))
    #affine = self.swap_affine_xy(affine)
    pts_aff[:, 0:2] = pts
    pts_out = np.zeros_like(pts)
    for i in range(pts.shape[0]):
        pts_out[i][0:2] = np.matmul(affine, pts_aff[i])[0:2]
    return pts_out


def gen_top_down_labels(path, affine, img_w, img_h, map_w, map_h, incl_path=True, incl_endp=True):
    seg_labels = np.zeros([img_w, img_h, 2]).astype(float)
    path_in_img = cf_to_img(path, np.array([map_w, map_h]))
    gauss_sigma = map_w / 96

    seg_labels[:, :, 0] = plot_path_on_img(seg_labels[:,:,0], path_in_img)
    if len(path_in_img) > 1:
        seg_labels[:,:,1] = plot_dot_on_img(seg_labels[:,:,1], path_in_img[-1], gauss_sigma)

    seg_labels_rot = apply_affine(seg_labels, affine, img_w, img_h)
    seg_labels_rot[:, :, 0] = gaussian_filter(seg_labels_rot[:, :, 0], gauss_sigma)
    seg_labels_rot[:, :, 1] = gaussian_filter(seg_labels_rot[:, :, 1], gauss_sigma)

    # Standardize both channels separately (each has mean zero, unit variance)
    seg_labels_path = standardize_2d_prob_dist(seg_labels_rot[:, :, 0:1])
    seg_labels_endpt = standardize_2d_prob_dist(seg_labels_rot[:, :, 1:2])

    if DEBUG:
        cv2.imshow("l_traj", seg_labels_path[0, :, :])
        cv2.imshow("l_endpt", seg_labels_endpt[0, :, :])
        cv2.waitKey(0)

    if incl_path and not incl_endp:
        seg_labels_rot = seg_labels_path
    elif incl_endp and not incl_path:
        seg_labels_rot = seg_labels_endpt
    else:
        seg_labels_rot = np.concatenate((seg_labels_path, seg_labels_endpt), axis=0)

    seg_labels_t = torch.from_numpy(seg_labels_rot).unsqueeze(0).float()
    return seg_labels_t


def gen_top_down_image(env_top_down_image, affine, img_w, img_h, map_w, map_h):
    #top_down_image = load_env_img(env_id)
    # TODO: Check for overflowz
    seg_img = env_top_down_image.copy()
    seg_img_rot = apply_affine(seg_img, affine, img_w, img_h)

    if DEBUG:
        cv2.imshow("rot_top", seg_img_rot)
        cv2.waitKey(10)

    #self.latest_rot_img_dbg = seg_img_rot

    seg_img_rot = standardize_image(seg_img_rot)
    seg_img_t = torch.from_numpy(seg_img_rot).unsqueeze(0).float()

    return seg_img_t


class TopDownDataset(Dataset):
    def __init__(self,
                 env_list=None,
                 instr_negatives=False,
                 instr_negatives_similar_only=False,
                 seg_level=False,
                 yaw_rand_range=0,
                 img_w=512,
                 img_h=512,
                 map_w=None,
                 map_h=None,
                 incl_path=True,
                 incl_endpoint=False,
                 use_semantic_maps=False):

        # If data is already loaded in memory, use it
        self.cuda = False
        self.env_list = env_list
        self.train_instr, self.dev_instr, self.test_instr, corpus = get_all_instructions()
        self.all_instr = {**self.train_instr, **self.dev_instr, **self.test_instr}
        self.token2term, self.word2token = get_word_to_token_map(corpus)
        self.thesaurus = load_landmark_alignments()
        self.include_instr_negatives = instr_negatives
        #if instr_negatives:
        #    self.similar_instruction_map = load_similar_instruction_map()
        self.instr_negatives_similar_only = instr_negatives_similar_only

        self.use_semantic_maps = use_semantic_maps

        self.img_w = img_w
        self.img_h = img_h

        if map_w is None:
            self.map_w = self.img_w
            self.map_h = self.img_h
        else:
            self.map_w = map_w
            self.map_h = map_h

        self.yaw_rand_range = yaw_rand_range
        self.latest_img_dbg = None
        self.latest_rot_img_dbg = None

        self.incl_endpoint = incl_endpoint
        self.incl_path = incl_path

        # If the data is supposed to be at seg level (not nested envs + segs), then we can support batching
        # but we need to correctly infer the dataset size
        self.seg_level = seg_level
        if seg_level:
            self.seg_list = []
            for env in self.env_list:
                for set_idx, set in enumerate(self.all_instr[env]):
                    for seg_idx, seg in enumerate(set["instructions"]):
                        self.seg_list.append([env, set_idx, seg_idx])

        print("Initialzied dataset!")
        print("   yaw range : " + str(self.yaw_rand_range))
        print("   map size: ", self.map_w, self.map_h)
        print("   img size: ", self.img_w, self.img_h)


    def __len__(self):
        return len(self.env_list) if not self.seg_level else len(self.seg_list)

    def gen_instruction(self, instruction):
        tok_instruction = tokenize_instruction(instruction, self.word2token)
        instruction_t = torch.LongTensor(tok_instruction)

        # If we're doing segment level, we want to support batching later on.
        # Otherwise each instance is a batch in itself
        # TODO Move unsqueezing into the collate_fn
        if not self.seg_level:
            instruction_t = instruction_t.unsqueeze(0)
        return instruction_t

    def gen_neg_instructions(self, env_id, seg_idx):
        # If we are to be using similar instructions according to the json file, then
        # initialize choices with similar instructions. Otherwise let choices be empty, and they will
        # be filled in the following lines.
        if self.instr_negatives_similar_only:
            choices = self.similar_instruction_map[str(env_id)][str(seg_idx)]
        else:
            choices = []
        # If there are no similar instructions to this instruction, pick a completely random instruction
        if len(choices) == 0:
            while len(choices) == 0:
                env_options = list(self.similar_instruction_map.keys())
                random_env = random.choice(env_options)
                seg_options = list(self.similar_instruction_map[random_env].keys())
                if len(seg_options) == 0:
                    continue
                random_seg = random.choice(seg_options)
                choices = self.similar_instruction_map[random_env][random_seg]

        pick = random.choice(choices)
        picked_env = pick["env_id"]
        picked_seg = pick["seg_idx"]
        picked_set = pick["set_idx"]
        picked_instruction = self.all_instr[picked_env][picked_set]["instructions"][picked_seg]["instruction"]
        tok_fake_instruction = tokenize_instruction(picked_instruction, self.word2token)
        return torch.LongTensor(tok_fake_instruction).unsqueeze(0)

    def gen_lm_aux_labels(self, env_id, instruction, affine):

        env_conf_json = load_env_config(env_id)
        landmark_names, landmark_indices, landmark_positions = get_landmark_locations_airsim(env_conf_json)
        landmark_pos_in_img = pos_m_to_px(np.asarray(landmark_positions)[:, 0:2], np.array([self.map_w, self.map_h]))
        landmark_pos_in_seg_img = apply_affine_on_pts(landmark_pos_in_img, affine)

        if False:
            plot_path_on_img(self.latest_img_dbg, landmark_pos_in_img)
            plot_path_on_img(self.latest_rot_img_dbg, landmark_pos_in_seg_img)
            cv2.imshow("img", self.latest_img_dbg)
            cv2.imshow("rot_img", self.latest_rot_img_dbg)
            cv2.waitKey(0)

        landmark_pos_t = torch.from_numpy(landmark_pos_in_seg_img).unsqueeze(0)
        landmark_indices_t = torch.LongTensor(landmark_indices).unsqueeze(0)

        mask1 = torch.gt(landmark_pos_t, 0)
        mask2 = torch.lt(landmark_pos_t, self.img_w)
        mask = mask1 * mask2
        mask = mask[:, :, 0] * mask[:, :, 1]
        mask = mask

        landmark_pos_t = torch.masked_select(landmark_pos_t, mask.unsqueeze(2).expand_as(landmark_pos_t)).view(
            [-1, 2])
        landmark_indices_t = torch.masked_select(landmark_indices_t, mask).view([-1])

        mentioned_names, mentioned_indices = get_mentioned_landmarks(self.thesaurus, instruction)
        mentioned_labels_t = empty_float_tensor(list(landmark_indices_t.size())).long()
        for i, landmark_idx_present in enumerate(landmark_indices_t):
            if landmark_idx_present in mentioned_indices:
                mentioned_labels_t[i] = 1

        if len(landmark_indices_t) > 0:
            aux_label = {
                "lm_pos": landmark_pos_t,
                "lm_indices": landmark_indices_t,
                "lm_mentioned": mentioned_labels_t,
                "lm_visible": mask,
            }
        else:
            aux_label = {
                "lm_pos": [],
                "lm_indices": [],
                "lm_mentioned": [],
                "lm_visible": []
            }
        return aux_label

    def get_item(self, env_id, set_idx, seg_idx):

        path = load_path(env_id)
        env_image = load_env_img(env_id, self.map_w, self.map_h)

        self.latest_img_dbg = env_image

        data = {
            "images": [],
            "instr": [],
            "traj_labels": [],
            "affines_g_to_s": [],
            "lm_pos": [],
            "lm_indices": [],
            "lm_mentioned": [],
            "lm_visible": [],
            "set_idx": [],
            "seg_idx": [],
            "env_id": []
        }

        if self.include_instr_negatives:
            data["neg_instr"] = []

        # Somehow load the instruction with the start and end indices for each of the N segments
        if self.seg_level:
            instruction_segments = [self.all_instr[env_id][set_idx]["instructions"][seg_idx]]
        else:
            instruction_segments = self.all_instr[env_id][0]["instructions"]

        for seg_idx, seg in enumerate(instruction_segments):
            start_idx = seg["start_idx"]
            end_idx = seg["end_idx"]
            instruction = seg["instruction"]
            start_pt, dir_yaw = get_start_pt_and_yaw(path, start_idx, self.map_w, self.map_h, self.yaw_rand_range)
            if start_pt is None:
                continue
            affine = get_affine_matrix(start_pt, dir_yaw, self.img_w, self.img_h)

            if DEBUG:
                env_image = self.latest_img_dbg
                print("Start Pt: ", start_pt)
                print("Start Yaw: ", dir_yaw)
                path_img = cf_to_img(path, [env_image.shape[0], env_image.shape[1]])
                seg_path = path_img[start_idx:end_idx]
                env_image = env_image.copy()
                plot_path_on_img(env_image, seg_path)

            seg_img_t = gen_top_down_image(env_image, affine, self.img_w, self.img_h, self.map_w, self.map_h)
            seg_labels_t = gen_top_down_labels(path[start_idx:end_idx], affine, self.img_w, self.img_h, self.map_w, self.map_h, self.incl_path, self.incl_endpoint)
            instruction_t = self.gen_instruction(instruction)
            aux_label = self.gen_lm_aux_labels(env_id, instruction, affine)

            if DEBUG:
                cv2.waitKey(0)

            if self.include_instr_negatives:
                neg_instruction_t = self.gen_neg_instructions(env_id, seg_idx)
                data["neg_instr"].append(neg_instruction_t)

            data["images"].append(seg_img_t)
            data["instr"].append(instruction_t)
            data["traj_labels"].append(seg_labels_t)
            data["affines_g_to_s"].append(affine)
            data["env_id"].append(env_id)
            data["set_idx"].append(set_idx)
            data["seg_idx"].append(seg_idx)
            data = dictlist_append(data, aux_label)

        return data

    @DeprecationWarning
    # TODO: Get rid of this. This functionality moved into aux_data_providers
    def get_top_down_image_env(self, env_id, egocentric=False):
        """
        To be called externally to retrieve a top-down environment image oriented with the start of the requested segment
        :param env_id:  environment id
        :return:
        """
        path = load_path(env_id)
        env_image_in = load_env_img(env_id, self.map_w, self.map_h)

        # If we need to return a bigger image resolution than we loaded
        if self.map_w != self.img_w or self.map_h != self.img_h:
            env_image = np.zeros([self.img_h, self.img_w, env_image_in.shape[2]])
            env_image[0:self.map_h, 0:self.map_w, :] = env_image_in
        else:
            env_image = env_image_in

        #path_img = cf_to_img(path, [env_image.shape[0], env_image.shape[1]])
        #self.plot_path_on_img(env_image, path_img)

        env_image = standardize_image(env_image)
        env_img_t = torch.from_numpy(env_image).unsqueeze(0).float()
        #presenter = Presenter()
        #presenter.show_image(env_img_t[0], "data_img", torch=True, scale=1)
        return env_img_t

    @DeprecationWarning
    # TODO: Probably get rid of this
    def get_top_down_image(self, env_id, set_idx, seg_idx):
        """
        To be called externally to retrieve a top-down environment image oriented with the start of the requested segment
        :param env_id:  environment id
        :param set_idx: instruction set number
        :param seg_idx: segment index
        :return:
        """
        # TODO: Revise the bazillion versions of poses - get rid of this specific one
        path = load_path(env_id)
        env_image = load_env_img(env_id, self.map_w, self.map_h)

        path_img = cf_to_img(path, [env_image.shape[0], env_image.shape[1]])
        plot_path_on_img(env_image, path_img)

        seg = self.all_instr[env_id][set_idx]["instructions"][seg_idx]

        start_idx = seg["start_idx"]
        start_pt, dir_yaw = get_start_pt_and_yaw(path, start_idx, self.map_w, self.map_h, self.yaw_rand_range)
        if start_pt is None:
            return None
        affine = get_affine_matrix(start_pt, dir_yaw)
        seg_img_t = self.gen_top_down_image(env_image, affine)
        #seg_img_t = seg_img_t.permute(0, 1, 3, 2)

        # A 2D pose is specified as [pos_x, pos_y, yaw]
        # A 3D pose would be [pos_x, pos_y, pos_z, r_x, r_y, r_z, r_w]
        img_pose_2d = {"pos": start_pt, "yaw": dir_yaw}
        img_pose_2d_t = torch.FloatTensor([start_pt[0], start_pt[1], dir_yaw]).unsqueeze(0)
        return seg_img_t, img_pose_2d_t

    def __getitem__(self, idx):
        if self.seg_level:
            env_id = self.seg_list[idx][0]
            set_idx = self.seg_list[idx][1]
            seg_idx = self.seg_list[idx][2]
        else:
            env_id = self.env_list[idx]
            set_idx = 0
            seg_idx = 0

        return self.get_item(env_id, set_idx, seg_idx)

    def set_word2token(self, token2term, word2token):
        self.token2term = token2term
        self.word2token = word2token

    def collate_one(self, one):
        one = torch.stack(one, dim=0)
        if self.cuda:
            one = one.cuda()
        one = Variable(one)
        return one

    def collate_fn(self, list_of_samples):
        if None in list_of_samples:
            return None

        if not self.seg_level:
            data = dict_zip(list_of_samples)
            return data
        else:
            # Keep only those samples that have data
            list_of_samples = [sample for sample in list_of_samples if len(sample["images"]) > 0]
            data = dict_zip(list_of_samples)
            # each is a list of lists, where the inner lists contain a single element each. Turn it into a list of elements
            data = dict_map(data, lambda m: [a[0] for a in m])#, ["images", "instr", "traj_labels", "lm_pos", "lm_idx", "lm_mentioned", "lm_visible"])

            if "images" not in data:
                return None

            # images and labels are not sequences and can be easily cat into a batch
            data["images"] = torch.cat(data["images"], 0)
            data["traj_labels"] = torch.cat(data["traj_labels"], 0)

            # Instructions are variable length, so we need to pad them in a tensor that has space for the longest instruction
            data["instr"], data["instr_mask"] = sequence_list_to_masked_tensor(data["instr"])

            # All the other things are sequences and we just leave them as lists. The model should sort it out.
            return data