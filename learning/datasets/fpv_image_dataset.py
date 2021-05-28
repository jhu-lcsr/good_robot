import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from torchvision.transforms.functional import to_tensor
from learning.inputs.pose import Pose
from learning.inputs.vision import standardize_image
from learning.models.semantic_map.pinhole_camera_inv import PinholeCameraProjection
from data_io.env import get_landmark_locations_airsim
from learning.datasets.fpv_data_augmentation import data_augmentation
from data_io.paths import get_poses_dir, get_fpv_img_flight_dir, load_config_files

from utils.simple_profiler import SimpleProfiler

import parameters.parameter_server as P

PROFILE = False

class FpvImageDataset(Dataset):
    def __init__(self, env_ids, dataset_name, eval, real, real_poses=None):
        self.prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        self.real = real
        if real_poses:
            self.real_poses = real_poses
        else:
            self.real_poses = real
        self.eval = eval
        self.dataset_name = dataset_name
        self.env_ids = env_ids

        # Assume that these parameters include cam_h_fov, img_w, img_h
        self.model_params = P.get_current_parameters()["Model"]
        self.cam_h_fov = self.model_params["cam_h_fov"]
        self.img_w = self.model_params["img_w"]
        self.img_h = self.model_params["img_h"]
        self.data_params = P.get_current_parameters()["Data"]
        self.load_img_w = self.data_params["load_img_w"]
        self.load_img_h = self.data_params["load_img_h"]

        self.prof.tick("out")
        self.instructions, self.poses, self.images, self.env_ids_decompressed = self.data_from_env_ids(env_ids)

        self.prof.tick("data from env")
        self.lm_pos_fpv, self.lm_idx, self.lm_pos_map = self.compute_pos_idx(add_null=0)
        self.prof.tick("compute pos idx")
        self.filter_none()
        self.prof.tick("filter none")

        self.update_dic()
        self.prof.tick("update dic")
        self.prof.print_stats()

    def __len__(self):
        return len(self.env_ids_decompressed)

    def __getitem__(self, index):
        prof = SimpleProfiler(torch_sync=PROFILE, print=PROFILE)
        prof.tick("out")
        if type(index) == int:
            image = self.images[index]
            lm_pos_fpv = self.lm_pos_fpv[index]
            lm_indices = self.lm_idx[index]
            lm_pos_map = self.lm_pos_map[index]
            prof.tick("retrieve data")

            # data augmentation. If eval no data augmentation.
            out_img, out_lm_indices, out_lm_pos_fpv = data_augmentation(
                image, lm_indices, lm_pos_fpv, self.img_h, self.img_w, self.eval, prof)
            if (len(out_lm_indices) == 0) | (out_lm_indices is None):
                out_img, out_lm_indices, out_lm_pos_fpv = data_augmentation(
                    image, lm_indices, lm_pos_fpv, self.img_h, self.img_w, True, prof)

            out_img = standardize_image(np.array(out_img))
            out_img = torch.from_numpy(out_img)

            out_lm_indices = torch.tensor(out_lm_indices)
            out_lm_pos_fpv = torch.tensor(out_lm_pos_fpv)

            sample = {"poses": self.poses[index],
                      "instructions": [],  # self.instructions[index],
                      "images": out_img,
                      "env_ids": self.env_ids_decompressed[index],
                      "lm_pos_fpv": out_lm_pos_fpv,
                      "lm_indices": out_lm_indices,
                      "lm_pos_map": lm_pos_map}
            prof.tick("dic")
            prof.print_stats()

        """
        elif type(index) == list:
            out_images_list, out_lm_indices_list, out_lm_pos_fpv_list = [], [], []
            for i in index:
                image = self.images[i]
                lm_pos_fpv = self.lm_pos_fpv[i]
                lm_indices = self.lm_idx[i]

                out_img, out_lm_indices, out_lm_pos_fpv = data_augmentation(image, lm_indices, lm_pos_fpv, IMG_HEIGHT, IMG_WIDTH, self.eval, prof)

                if (len(out_lm_indices) == 0) | (out_lm_indices is None):
                    out_img, out_lm_indices, out_lm_pos_fpv = data_augmentation(image, lm_indices, lm_pos_fpv, IMG_HEIGHT, IMG_WIDTH, True, prof)

                out_images_list.append(out_img)
                out_lm_indices_list.append(out_lm_indices)
                out_lm_pos_fpv_list.append(out_lm_pos_fpv)

            sample = {"poses": [self.poses[i] for i in index],
                      "instructions": [],  # self.instructions[index],
                      "lm_mentioned": [],
                      "images": out_images_list,
                      "env_ids": [self.env_ids_decompressed[i] for i in index],
                      "lm_pos_fpv": out_lm_pos_fpv_list,
                      "lm_idx": out_lm_indices_list}
        """
        return sample


    def data_from_env_ids(self, env_ids, proba_selection=1.0):
        images = []
        poses = []
        # list of all env_ids (with duplicates)
        env_ids_decompressed = []
        # TODO: fill instructions
        instructions = []
        print("Using {} images".format("real" if self.real else "simulated"))
        for env_id in env_ids:
            poses_dir = get_poses_dir(env_id)
            images_dir = get_fpv_img_flight_dir(env_id, self.real)
            pose_filenames = [f for f in os.listdir(poses_dir) if f.endswith('.json')]
            image_filenames = [f for f in os.listdir(images_dir) if (f.endswith('.jpg') | f.endswith('.png'))]
            try:
                assert len(image_filenames) == len(pose_filenames)
            except:
                print("error {}: different count of poses and images".format(env_id))

            if not(os.listdir(images_dir)):
                print(images_dir+"is empty")
                assert(not(not(os.listdir(images_dir))))

            img_ids = np.sort(
                [int(f.replace('.', '_').split('_')[-2]) for f in os.listdir(images_dir) if (f.endswith('.jpg') | f.endswith('.png'))])
            try:
                selected = np.random.choice(img_ids,
                                            int(len(image_filenames) * proba_selection),
                                            replace=False)
            except:
                print(img_ids)
            selected_ids = np.sort(selected)

            for img_id in selected_ids:
                filename_pose = "pose_{}.json".format(img_id)

                gen_imgpath = lambda id,ext: os.path.join(images_dir, f"usb_cam_{img_id}.{ext}")
                img_path = gen_imgpath(img_id, "jpg")
                if not os.path.exists(img_path):
                    img_path = gen_imgpath(img_id, "png")

                #print(filename_pose, filename_img)
                with open(os.path.join(poses_dir, filename_pose), 'r') as f:
                    try:
                        pose = json.load(f)["camera"]
                        poses.append(pose)
                        read_success = True
                    except:
                        read_success = False

                if read_success:
                    # Images are resized in bigger shape. They will be resized to 256*144 after data augmentation
                    img = Image.open(img_path).resize((self.load_img_w, self.load_img_h))
                    images.append(img)

                    env_ids_decompressed.append((env_id, img_id))

        return instructions, poses, images, env_ids_decompressed

    def update_dic(self):
        self.dic = {"poses": self.poses,
               "instructions": self.instructions,
               "images": self.images,
               "env_ids": self.env_ids_decompressed,
               "lm_pos_fpv": self.lm_pos_fpv,
               "lm_indices": self.lm_idx,
                "lm_pos_map": self.lm_pos_map}

    def provider_lm_pos_lm_indices_fpv(self, env_ids, add_null=0):
        """
        Data provider that gives the positions and indices of all landmarks visible in the FPV image.
        :param pose_list: B*7 list of poses decomposed in 3 position and 4 orientation floats
         [x,y,z, orient_x, orient_y, orient_z, orient_w]
         img_x, img_y: shape of images
         env_ids: list of environments.
        :return: ("lm_pos", lm_pos) - lm_pos is a list (over timesteps) of lists (over landmarks visible in image) of the
                    landmark locations in image pixel coordinates
                 ("lm_indices", lm_indices) - lm_indices is a list (over timesteps) of lists (over landmarks visible in image)
                    of the landmark indices for every landmark included in lm_pos. These are the landmark classifier labels
        """
        list_of_conf = load_config_files(np.unique(env_ids))#, perception=True)
        # add add_null empty objects on each config.
        if add_null > 0:
            for i, conf in enumerate(list_of_conf):
                zpos = conf["zPos"]
                xpos = conf["xPos"]
                lm_positions = np.stack([xpos, zpos], 1)
                for _ in range(add_null):  # add 2 empty objects on configuration
                    i_null = 0
                    while i_null < 100:
                        xnull = np.random.rand() * 4.7
                        znull = np.random.rand() * 4.7
                        distances_to_lm = np.linalg.norm(lm_positions - np.array([xnull, znull]), axis=1)
                        min_dist_to_lm = np.min(distances_to_lm)
                        if min_dist_to_lm > 1.2:
                            break
                        i_null += 1

                    list_of_conf[i]["xPos"].append(xnull)
                    list_of_conf[i]["zPos"].append(znull)
                    list_of_conf[i]["landmarkName"].append("0Null")
                    list_of_conf[i]["radius"].append("100")

        landmark_indices_list = []
        landmark_pos_list = []
        for conf_json in list_of_conf:
            lm_names, landmark_indices, landmark_pos = get_landmark_locations_airsim(conf_json, add_empty=True)
            #landmark_pos = get_landmark_locations(conf_json)
            landmark_indices_list.append(landmark_indices)
            landmark_pos_list.append(landmark_pos)

        # TODO: Grab image size from segment_data

        # TODO: recode CAM_FOV in parameters instead of hardcoding
        projector = PinholeCameraProjection(
            map_size_px=None,
            world_size_px=None,
            world_size_m=None,
            img_x=self.load_img_w,
            img_y=self.load_img_h,
            cam_fov=self.cam_h_fov,
            use_depth=False,
            start_height_offset=0.0)
        n_obs = len(self.poses)

        lm_pos_fpv = []
        lm_indices = []
        lm_mentioned = []
        lm_pos_map = []

        for i_obs in range(n_obs):

            # index of the environment in the list of unique environments
            env_id = env_ids[i_obs]
            i_env_id = np.where(np.unique(env_ids) == env_id)[0][0]

            t_lm_pos_fpv = []
            t_lm_indices = []
            t_lm_pos_map = []

            if self.poses[i_obs] is not None:
                cam_pos = self.poses[i_obs]['position']
                cam_rot = self.poses[i_obs]['orientation']
                # convert xyzw to wxyz (airsim convention)
                cam_rot_airsim = [cam_rot[-1]] + cam_rot[:-1]

                for i_lm, landmark_in_world in enumerate(landmark_pos_list[i_env_id]):
                    # landmark_in_world = landmark_in_world[0]
                    landmark_idx = landmark_indices_list[i_env_id][i_lm]

                    landmark_in_img, landmark_in_cam, status = projector.world_point_to_image(cam_pos, cam_rot_airsim,
                                                                                              landmark_in_world)
                    # This is None if the landmark is behind the camera.
                    if landmark_in_img is not None:
                        # presenter.save_image(images[timestep], name="tmp.png", torch=True, draw_point=landmark_in_img)
                        t_lm_pos_fpv.append(landmark_in_img[0:2])
                        t_lm_pos_map.append(landmark_in_world[0:2])
                        t_lm_indices.append(landmark_idx)
                        # t_lm_mentioned.append(this_lm_mentioned)

            if len(t_lm_pos_fpv) > 0:

                t_lm_pos_fpv = torch.from_numpy(np.asarray(t_lm_pos_fpv)).float()
                t_lm_pos_map = torch.from_numpy(np.asarray(t_lm_pos_map)).float()
                t_lm_indices = torch.from_numpy(np.asarray(t_lm_indices)).long()

            else:
                t_lm_pos_fpv = None
                t_lm_pos_map = None
                t_lm_indices = None
                t_lm_mentioned = None

            lm_pos_fpv.append(t_lm_pos_fpv)
            lm_pos_map.append(t_lm_pos_map)
            lm_indices.append(t_lm_indices)
            # lm_mentioned.append(t_lm_mentioned)

        return np.array(lm_pos_fpv), np.array(lm_indices), lm_pos_map

    def compute_pos_idx(self, add_null=0): # number of Null objects added to the map
        """

        :param add_null: How many empty objects are added per config. 1 is generally enough
        :return: landmark positions on images, landmark imdices on mages, lanmdark coordinates on map.
        """
        env_ids = [x[0] for x in self.env_ids_decompressed]
        # Provider is inspired from provider used for Airsim but different
        lm_pos_fpv, lm_idx_fpv, lm_pos_map = self.provider_lm_pos_lm_indices_fpv(env_ids, add_null)
        return lm_pos_fpv, lm_idx_fpv, lm_pos_map

    def filter_none(self):
        # Filter images that contain no object

        no_none = []
        for i, idx_list in enumerate(self.lm_idx):
            if not((idx_list is None)):
                if len(idx_list) > 0:
                    no_none.append(i)

        self.poses = [self.poses[i] for i in no_none]
        self.images = [self.images[i] for i in no_none]
        self.env_ids_decompressed = [self.env_ids_decompressed[i] for i in no_none]
        self.lm_idx = [self.lm_idx[i] for i in no_none]
        self.lm_pos_fpv = [self.lm_pos_fpv[i] for i in no_none]
        self.lm_pos_map = [self.lm_pos_map[i] for i in no_none]

    def dic_to_pose(self, dic_of_pose):
        """
        :param dic_of_pose: pose stored as a dictionary
        :return: Pose object
        """
        pose = Pose(torch.tensor(dic_of_pose['position']), torch.tensor(dic_of_pose['orientation']))
        return pose

    def collate_fn(self, list_of_samples):
        images = [sample['images'] for sample in list_of_samples]
        lm_indices = [sample['lm_indices'] for sample in list_of_samples]
        lm_pos_fpv = [sample['lm_pos_fpv'] for sample in list_of_samples]
        lm_pos_map = [sample['lm_pos_map'] for sample in list_of_samples]
        env_ids = [sample['env_ids'] for sample in list_of_samples]
        poses = [self.dic_to_pose(sample['poses'])for sample in list_of_samples]

        images = torch.stack(images, dim=0)

        keys = ["images", "env_ids", "poses", "lm_indices", "lm_pos_fpv", "lm_pos_map"] #"instructions", "lm_mentioned"]
        out_tuple = (images,  env_ids, poses, lm_indices, lm_pos_fpv, lm_pos_map)  # instructions, lm_mentioned)
        out_dict = dict(zip(keys, out_tuple))
        return out_dict


def get_stats(b):
    idx_a = [np.array(x) for x in b["labels"]]
    flatten = [x for sublist in idx_a for x in sublist]
    values, counts = np.unique(flatten, return_counts=True)
    return values, counts


def get_stats_total(batch):
    dic_out = {"real": {}, "sim": {}}
    if len(batch) == 2:
        values_real, counts_real = get_stats(batch["real"])
        values_sim, counts_sim = get_stats(batch["sim"])
        for i, v in enumerate(values_real):
            dic_out["real"][str(v)] = counts_real[i]
        for i, v in enumerate(values_sim):
            dic_out["sim"][str(v)] = counts_sim[i]
    else:
        values, counts = get_stats(batch)
        dic_out = dict(zip(values, counts))
    return dic_out