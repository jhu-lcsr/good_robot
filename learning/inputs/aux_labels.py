import torch
import numpy as np

from data_io.env import load_env_config, load_template, load_path, get_landmark_locations_airsim
from data_io.instructions import load_landmark_alignments, get_all_instructions, \
    get_word_to_token_map, get_mentioned_landmarks
from env_config.definitions.landmarks import get_landmark_name_to_index
from data_io.units import UnrealUnits

from learning.models.semantic_map.pinhole_projection_map import PinholeProjector


class AuxLabelsBase:
    """
    Base-class for auxiliary label generation.
    Provides all the auxiliary labels that are in common between natural language and templated language datasets:
        * Landmark classification labels
        * Goal location labels
    Subclasses of this (below) should provide:
        * Relevant landmark labels (grounding)
        * Language understanding labels
    """
    def __init__(self):
        self.side_indices = {"left": 0, "right": 1}

    def __call__(self, images, states, segment_data, mask):
        projector = PinholeProjector(img_x=images.size(3), img_y=images.size(2))
        # presenter = Presenter()

        env_id = segment_data.metadata[0]["env_id"]

        conf_json = load_env_config(env_id)
        all_landmark_indices = get_landmark_name_to_index()
        landmark_names, landmark_indices, landmark_pos = get_landmark_locations_airsim(conf_json)

        path_array = load_path(env_id)
        goal_loc = self.__get_goal_location_airsim(path_array)

        # Traj length x 64 landmarks x 14
        # 0-5: Present landmarks data
        #   0 - landmark present in img
        #   1-2 - landmark pix_x | pix_y
        #   3-5 - landmark world coords m_x | m_y
        # 6-7: Template data
        #   6 - landmark_mentioned index
        #   7 - mentioned_side index
        #   8 - landmark mentioned
        # 9-13: Goal data
        #   9-10 - goal_x_pix | goal_y_pix
        #   11-12 - goal_x | goal_y (world)
        #   13 - goal visible
        aux_labels = torch.zeros((images.size(0), len(all_landmark_indices), 14))

        # Store goal location in airsim coordinates
        aux_labels[:, :, 11:13] = torch.from_numpy(goal_loc[0:2]).unsqueeze(0).unsqueeze(0).expand_as(
            aux_labels[:, :, 11:13])

        for i, idx in enumerate(landmark_indices):
            aux_labels[:, idx, 3:6] = torch.from_numpy(
                landmark_pos[i]).unsqueeze(0).clone().repeat(aux_labels.size(0), 1, 1)

        for timestep in range(images.size(0)):
            # presenter.save_image(images[timestep], name="tmp.png", torch=True)

            if mask[timestep] == 0:
                continue

            cam_pos = states[timestep, 9:12]
            cam_rot = states[timestep, 12:16]

            goal_in_img, goal_in_cam, status = projector.world_point_to_image(cam_pos, cam_rot, goal_loc)
            if goal_in_img is not None:
                aux_labels[timestep, :, 9:11] = torch.from_numpy(goal_in_img[0:2]).unsqueeze(0).expand_as(
                    aux_labels[timestep, :, 9:11])
                aux_labels[timestep, :, 13] = 1.0

            for i, landmark_world in enumerate(landmark_pos):
                landmark_idx = landmark_indices[i]

                landmark_in_img, landmark_in_cam, status = projector.world_point_to_image(cam_pos, cam_rot,
                                                                                          landmark_world)
                # This is None if the landmark is behind the camera.
                if landmark_in_img is not None:
                    # presenter.save_image(images[timestep], name="tmp.png", torch=True, draw_point=landmark_in_img)
                    aux_labels[timestep, landmark_idx, 0] = 1.0
                    aux_labels[timestep, landmark_idx, 1:3] = torch.from_numpy(landmark_in_img[0:2])
                    # aux_labels[timestep, landmark_idx, 3:6] = torch.from_numpy(landmark_in_cam[0:3])
                    # aux_labels[timestep, landmark_idx, 8] = 1.0 if landmark_idx == mentioned_landmark_idx else 0

        return aux_labels

    def __get_goal_location_airsim(self, path_array):
        units = UnrealUnits(1.0)
        goal_x = path_array[-1][0]
        goal_y = path_array[-1][1]
        pt = np.asarray([goal_x, goal_y])
        pt_as = np.zeros(3)
        pt_as[0:2] = units.pos2d_to_as(pt)
        return pt_as

class AuxLabelsNL(AuxLabelsBase):
    def __init__(self):
        super(AuxLabelsNL, self).__init__()
        self.thesaurus = load_landmark_alignments()
        train_instructions, dev_instructions, test_instructions, corpus = get_all_instructions()
        self.all_instructions = {**train_instructions, **dev_instructions, **test_instructions}
        self.corpus = corpus
        self.word2token, self.token2term = get_word_to_token_map(corpus)

    def __call__(self, images, states, segment_data, mask):
        aux_labels = super(AuxLabelsNL, self).__call__(images, states, segment_data, mask)
        env_id = segment_data.metadata[0]["env_id"]
        set_idx = segment_data.metadata[0]["set_idx"]
        seg_idx = segment_data.metadata[0]["seg_idx"]

        str_instruction = self.all_instructions[env_id][set_idx]["instructions"][seg_idx]["instruction"]

        added = self.__add_aux_data_nl(aux_labels, str_instruction)
        if not added:
            print ("Couldn't add auxiliary NL data!")
        return aux_labels

    def __add_aux_data_nl(self, labels, str_instruction):

        mentioned_landmark_names, mentioned_landmark_indices = get_mentioned_landmarks(self.thesaurus, str_instruction)

        if len(mentioned_landmark_indices) > 0:
            labels[:, :, 6] = mentioned_landmark_indices[0]
        for landmark_idx in range(labels.size(1)):
            labels[:, landmark_idx, 8] = 1.0 if landmark_idx in mentioned_landmark_indices else 0

        return True


class AuxLabelsTemplateLandmarkSide(AuxLabelsBase):

    def __init__(self):
        super(AuxLabelsTemplateLandmarkSide, self).__init__()

    def __call__(self, images, states, segment_split, mask):
        aux_labels = super(AuxLabelsTemplateLandmarkSide, self).__call__(images, states, segment_split, mask)
        env_id = segment_split.metadata[0]["env_id"]
        added = self.__add_aux_data_template(aux_labels, env_id)
        if not added:
            print ("Couldn't add auxiliary template data!")
        return aux_labels

    def __add_aux_data_template(self, labels, env_id):
        template_json = load_template(env_id)

        # Template data is unavailable - means this is probably real natural language
        if template_json is None:
            return False

        mentioned_landmark_idx, mentioned_side_idx = self.__get_goal_landmark_idx(template_json)

        labels[:, :, 6] = mentioned_landmark_idx
        labels[:, :, 7] = mentioned_side_idx

        for landmark_idx in range(labels.size(1)):
            labels[:, landmark_idx, 8] = 1.0 if landmark_idx == mentioned_landmark_idx else 0

        return True

    def __get_goal_landmark_idx(self, template_json):
        landmark_name = template_json["landmark1"]
        landmark_indices = get_landmark_name_to_index()
        idx = landmark_indices[landmark_name]
        side = template_json["side"]
        side_idx = self.side_indices[side]
        return idx, side_idx


