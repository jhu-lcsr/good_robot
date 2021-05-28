from collections import namedtuple
from geometry import vec_to_yaw
from transformations import get_affine_rot_2d, get_affine_trans_2d, get_affine_scale_2d
import skimage.draw as draw
import skimage.transform as transform

import numpy as np
import random
from scipy.ndimage.filters import gaussian_filter
import torch
import cv2
from torch.autograd import Variable
from torch.utils.data import Dataset
from utils.dict_tools import dict_zip

from data_io.instructions import tokenize_instruction, get_all_instructions, get_word_to_token_map, \
    load_landmark_alignments, get_mentioned_landmarks
from data_io.meta import load_similar_instruction_map
from data_io.env import load_env_img, load_path, load_env_config, get_landmark_locations_airsim
from learning.inputs.vision import standardize_image
from learning.inputs.common import empty_float_tensor


class TopDownDatasetSM(Dataset):
    def __init__(self,
                 env_list=None,
                 instr_negatives=False,
                 instr_negatives_similar_only=False,
                 seg_level=False,
                 img_scale=1,
                 yaw_rand_range=0,
                 pos_rand_range=0
                 ):
        # If data is already loaded in memory, use it
        self.env_list = env_list
        self.train_instr, self.dev_instr, self.test_instr, corpus = get_all_instructions()
        self.all_instr = {**self.train_instr, **self.dev_instr, **self.test_instr}
        self.token2term, self.word2token = get_word_to_token_map(corpus)
        self.thesaurus = load_landmark_alignments()
        self.include_instr_negatives = instr_negatives
        if instr_negatives:
            self.similar_instruction_map = load_similar_instruction_map()
        self.instr_negatives_similar_only = instr_negatives_similar_only
        self.img_scale = img_scale

        self.yaw_rand_range = yaw_rand_range
        self.pos_rand_range = pos_rand_range
        self.pos_rand_image = 0

        # If the data is supposed to be at seg level (not nested envs + segs), then we can support batching
        # but we need to correctly infer the dataset size
        self.seg_level = seg_level
        if seg_level:
            self.seg_list = []
            for env in self.env_list:
                for set_idx, set in enumerate(self.all_instr[env]):
                    for seg_idx, seg in enumerate(set["instructions"]):
                        self.seg_list.append([env, set_idx, seg_idx])


    def __len__(self):
        return len(self.env_list) if not self.seg_level else len(self.seg_list)


    def get_affine_matrix(self, path, start_idx, origin, world_scaling_factor):
        if start_idx > len(path) - 2:
            return None, None

        start_pt = path[start_idx]
        next_pt = path[start_idx + 1]
        dir_vec = next_pt - start_pt
        dir_yaw = vec_to_yaw(dir_vec)

        if self.yaw_rand_range > 0:
            dir_yaw_offset = random.uniform(-self.yaw_rand_range, self.yaw_rand_range)
            dir_yaw += dir_yaw_offset

        if self.pos_rand_image > 0:
            pos_offset = random.uniform(0, self.pos_rand_range)
            angle = random.uniform(-np.pi, np.pi)
            offset_vec = pos_offset * np.array([np.cos(angle), np.sin(angle)])
            start_pt += offset_vec

        affine_s = get_affine_scale_2d([world_scaling_factor, world_scaling_factor])
        affine_t = get_affine_trans_2d(-start_pt)
        affine_rot = get_affine_rot_2d(-dir_yaw)
        affine_t2 = get_affine_trans_2d(origin)

        #return affine_t
        affine_total = np.dot(affine_t2, np.dot(affine_s, np.dot(affine_rot, affine_t)))
        out_crop_size = tuple(np.asarray(origin) * 2)

        return affine_total, out_crop_size

    def plot_path_on_img(self, img, path):
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

    def cf_to_img(self, img_size, as_coords):
        scale = img_size / 1000
        out_coords = as_coords * scale
        out_coords = img_size - out_coords
        return out_coords

    def as_to_img(self, img_size, as_coords):
        scale = img_size / 30
        out_coords = as_coords * scale
        out_coords[:, 0] = img_size - out_coords[:, 0]
        return out_coords

    def normalize_0_1(self, img):
        img = img - np.min(img)
        img = img / (np.max(img) + 1e-9)
        return img

    def apply_affine(self, img, affine_mat, out_crop_size):
        affine_swap = affine_mat[[1, 0]]
        affine_swap = affine_swap[:, [1,0,2]]
        out = cv2.warpAffine(img, affine_swap, out_crop_size)
        if len(out.shape) < len(img.shape):
            out = np.expand_dims(out, 2)
        return out

    def apply_affine_on_pts(self, pts, affine):
        pts_aff = np.ones((pts.shape[0], 3))
        pts_aff[:, 0:2] = pts
        pts_out = np.zeros_like(pts)
        for i in range(pts.shape[0]):
            pts_out[i][0:2] = np.matmul(affine, pts_aff[i])[0:2]
        return pts_out

    def __getitem__(self, idx):
        if self.seg_level:
            env_id = self.seg_list[idx][0]
            set_idx = self.seg_list[idx][1]
            seg_idx = self.seg_list[idx][2]
        else:
            env_id = self.env_list[idx]

        env_conf_json = load_env_config(env_id)
        landmark_names, landmark_indices, landmark_positions = get_landmark_locations_airsim(env_conf_json)

        top_down_image = load_env_img(env_id)

        path = load_path(env_id)

        img_x = top_down_image.shape[0]
        img_y = top_down_image.shape[1]

        path_in_img_coords = self.cf_to_img(img_x, path)
        landmark_pos_in_img = self.as_to_img(img_x, np.asarray(landmark_positions)[:, 0:2])
        self.pos_rand_image = self.pos_rand_range * img_x

        #self.plot_path_on_img(top_down_image, path_in_img_coords)
        #self.plot_path_on_img(top_down_image, landmark_pos_in_img)
        #cv2.imshow("top_down", top_down_image)
        #cv2.waitKey()

        input_images = []
        input_instructions = []
        label_images = []
        aux_labels = []

        # Somehow load the instruction with the start and end indices for each of the N segments
        if self.seg_level:
            instruction_segments = [self.all_instr[env_id][set_idx]["instructions"][seg_idx]]
        else:
            instruction_segments = self.all_instr[env_id][0]["instructions"]

        for seg_idx, seg in enumerate(instruction_segments):
            start_idx = seg["start_idx"]
            end_idx = seg["end_idx"]
            instruction = seg["instruction"]

            # TODO: Check for overflowz
            seg_path = path_in_img_coords[start_idx:end_idx]
            seg_img = top_down_image.copy()

            #test_plot = self.plot_path_on_img(seg_img, seg_path)
            # TODO: Validate the 0.5 choice, should it be 2?
            affine, cropsize = self.get_affine_matrix(seg_path, 0, [int(img_x / 2), int(img_y / 2)], 0.5)
            if affine is None:
                continue
            seg_img_rot = self.apply_affine(seg_img, affine, cropsize)

            seg_labels = np.zeros_like(seg_img[:, :, 0:1]).astype(float)
            seg_labels = self.plot_path_on_img(seg_labels, seg_path)
            seg_labels = gaussian_filter(seg_labels, 4)
            seg_labels_rot = self.apply_affine(seg_labels, affine, cropsize)

            #seg_labels_rot = gaussian_filter(seg_labels_rot, 4)
            seg_labels_rot = self.normalize_0_1(seg_labels_rot)

            # Change to true to visualize the paths / labels
            if False:
                cv2.imshow("rot_img", seg_img_rot)
                cv2.imshow("seg_labels", seg_labels_rot)
                rot_viz = seg_img_rot.astype(np.float64) / 512
                rot_viz[:, :, 0] += seg_labels_rot.squeeze()
                cv2.imshow("rot_viz", rot_viz)
                cv2.waitKey(0)

            tok_instruction = tokenize_instruction(instruction, self.word2token)
            instruction_t = torch.LongTensor(tok_instruction).unsqueeze(0)

            # Get landmark classification labels
            landmark_pos_in_seg_img = self.apply_affine_on_pts(landmark_pos_in_img, affine)

            # Down-size images and labels if requested by the model
            if self.img_scale != 1.0:
                seg_img_rot = transform.resize(
                    seg_img_rot,
                    [seg_img_rot.shape[0] * self.img_scale,
                     seg_img_rot.shape[1] * self.img_scale], mode="constant")
                seg_labels_rot = transform.resize(
                    seg_labels_rot,
                    [seg_labels_rot.shape[0] * self.img_scale,
                     seg_labels_rot.shape[1] * self.img_scale], mode="constant")
                landmark_pos_in_seg_img = landmark_pos_in_seg_img * self.img_scale

            seg_img_rot = standardize_image(seg_img_rot)
            seg_labels_rot = standardize_image(seg_labels_rot)
            seg_img_t = torch.from_numpy(seg_img_rot).unsqueeze(0).float()
            seg_labels_t = torch.from_numpy(seg_labels_rot).unsqueeze(0).float()

            landmark_pos_t = torch.from_numpy(landmark_pos_in_seg_img).unsqueeze(0)
            landmark_indices_t = torch.LongTensor(landmark_indices).unsqueeze(0)

            mask1 = torch.gt(landmark_pos_t, 0)
            mask2 = torch.lt(landmark_pos_t, seg_img_t.size(2))
            mask = mask1 * mask2
            mask = mask[:, :, 0] * mask[:, :, 1]
            mask = mask

            landmark_pos_t = torch.masked_select(landmark_pos_t, mask.unsqueeze(2).expand_as(landmark_pos_t)).view([-1, 2])
            landmark_indices_t = torch.masked_select(landmark_indices_t, mask).view([-1])

            mentioned_names, mentioned_indices = get_mentioned_landmarks(self.thesaurus, instruction)
            mentioned_labels_t = empty_float_tensor(list(landmark_indices_t.size())).long()
            for i, landmark_idx_present in enumerate(landmark_indices_t):
                if landmark_idx_present in mentioned_indices:
                    mentioned_labels_t[i] = 1

            aux_label = {
                "landmark_pos": landmark_pos_t,
                "landmark_indices": landmark_indices_t,
                "landmark_mentioned": mentioned_labels_t,
                "visible_mask": mask,
            }

            if self.include_instr_negatives:
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
                aux_label["negative_instruction"] = torch.LongTensor(tok_fake_instruction).unsqueeze(0)

            input_images.append(seg_img_t)
            input_instructions.append(instruction_t)
            label_images.append(seg_labels_t)
            aux_labels.append(aux_label)

        return [input_images, input_instructions, label_images, aux_labels]

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
            images, instructions, labels, aux_labels = tuple(zip(*list_of_samples))
            return [images, instructions, labels, aux_labels]
        else:
            # Keep only those samples that have data
            list_of_samples = [sample for sample in list_of_samples if len(sample[0]) > 0]
            images, instructions, labels, aux_labels = tuple(zip(*list_of_samples))
            # each is a list of lists, where the inner lists contain a single element each. Turn it into a list of elements
            images, instructions, labels, aux_labels = \
                map(lambda m: [a[0] for a in m], [images, instructions, labels, aux_labels])
            # Now images and labels can be trivially cat together, but aux_labels is a list of dicts.
            # Turn it into a dict of lists, where each can be trivially cat together
            aux_labels = dict_zip(aux_labels)

            # images and labels are not sequences and can be easily cat into a batch
            images = torch.cat(images, 0)
            labels = torch.cat(labels, 0)

            # All the other things are sequences and we just leave them as lists. The model should sort it out.
            return [images, instructions, labels, aux_labels]