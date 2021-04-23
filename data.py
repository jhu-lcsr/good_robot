import os 
import sys
import re
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from spacy.tokenizer import Tokenizer 
from spacy.lang.en import English
import torch
from torch.nn import functional as F
import cv2
import pdb 
import copy 
import pathlib 
import pickle as pkl
from tqdm import tqdm 

from annotate_data import Pair, flip_pair, rotate_pair, gaussian_augment 
np.random.seed(12) 
torch.manual_seed(12) 

PAD = "<PAD>"

class BaseTrajectory:
    """
    Trajectories are basic data-loading unit,
    provide access to all commands with their corresponding previous and next
    positional images/tuples, as well as metadata. 
    """
    def __init__(self,
                 line_id: int, 
                 commands: List,
                 previous_positions: List[Tuple[float]],
                 previous_rotations: List[Tuple[float]],
                 next_positions: List[Tuple[float]], 
                 next_rotations: List[Tuple[float]], 
                 images: List[str],
                 lengths: List[int],
                 traj_type: str = "flat",
                 batch_size: int = None,
                 resolution: int = 64,
                 do_filter: bool = False,
                 do_one_hot: bool = True,
                 top_only: bool = True, 
                 binarize_blocks: bool = False,
                 image_path: str = None):

        if batch_size is None:
            batch_size = len(commands)
        self.batch_size = batch_size 
        self.do_filter = do_filter
        self.do_one_hot = do_one_hot
        self.top_only = top_only
        self.binarize_blocks = binarize_blocks
        self.resolution = resolution
        self.image_path = image_path
        if self.image_path is not None:
            self.image_path = pathlib.Path(self.image_path)

        self.block_size = int((4 * self.resolution)/64)

        self.line_id = line_id
        self.commands = commands

        self.blocks_to_move = self.get_blocks_to_move(previous_positions, next_positions).float() 

        # output for previous positions has depth that gets filtered 
        self.next_positions_for_regression = next_positions
    
        self.previous_positions_for_pred = self.make_3d_positions(previous_positions, make_z = True, batch_size = self.batch_size)
        self.previous_positions_for_acc = copy.deepcopy(self.previous_positions_for_pred) 
        self.next_positions_for_pred = self.make_3d_positions(next_positions, make_z = True, batch_size = self.batch_size)
        self.next_positions_for_acc = copy.deepcopy(self.next_positions_for_pred) 
        #self.blocks_to_move = self.get_blocks_to_move()

        self.previous_positions = previous_positions
        self.next_positions = next_positions
        self.previous_rotations = previous_rotations
        self.next_rotations = next_rotations
        self.images = images

        if self.top_only:
            (self.previous_positions_for_pred, 
            self.previous_positions_for_acc,
            self.next_positions_for_pred, 
            self.next_positions_for_acc) = self.filter_top_only(self.previous_positions_for_pred, 
                                                                self.previous_positions_for_acc, 
                                                                self.next_positions_for_pred, 
                                                                self.next_positions_for_acc)

        self.previous_positions_input = self.get_input_positions()
       
        if self.do_filter: 
            # for loss, only look at the single block moved 
            self.previous_positions_for_pred = self.filter_by_blocks_to_move(self.previous_positions_for_pred)
            self.next_positions_for_pred = self.filter_by_blocks_to_move(self.next_positions_for_pred)

        self.lengths = lengths
        self.traj_vocab = set() 
        self.traj_type = traj_type


    def get_input_positions(self):
         # set as input but one-hot 
        if self.do_one_hot:
            previous_positions_input = self.one_hot(copy.deepcopy(self.previous_positions_for_acc)) 
        else:
            previous_positions_input = [x.reshape(-1, 1, self.resolution, self.resolution) for x in copy.deepcopy(self.previous_positions_for_acc)]
        return previous_positions_input


    def one_hot_helper(self, input_data):
        data = input_data.long()
        data = F.one_hot(data, 21)
        data = data.squeeze(3).squeeze(3)
        data = data.permute(0, 3, 1, 2)
        # make sure it is actuall OH 
        assert(torch.allclose(torch.sum(data, dim =1), torch.tensor(1).to(data.device)))
        data = data.float() 

        return data 

    def one_hot(self, input_positions): 
        data = [self.one_hot_helper(x) for x in input_positions]
        return data 

    def filter_by_blocks_to_move(self, positions):
        # zero out positions that aren't the block of interest
        for i in range(len(positions)):
            pos = positions[i]
            bidx = self.blocks_to_move[i]
            pos[pos != bidx] = 0 
            if self.binarize_blocks:
                pos[pos == bidx] = 1
            positions[i] = pos
        return positions 

    def filter_top_only(self, *data_list):
        # get top-down view from the states 
        to_ret = []
        # iterate over inputs 
        data_list = list(data_list)
        for i, data in enumerate(data_list):
            new_data = [torch.zeros((1, self.resolution, self.resolution, 1, 1)) for __ in range(len(data))]
            # iterate over steps in trajectory 
            for j, step_data in enumerate(data):
                for depth in range(6, -1, -1):
                    depth_slice = step_data[:,:,:,depth,:].unsqueeze(4)
                    # only add if new_data == 0
                    depth_slice[new_data[j] != 0] = 0
                    new_data[j] += depth_slice  
            # replace 
            data_list[i] = new_data
        return data_list 
        
    def get_blocks_to_move(self, prev_state, next_state):

        TOL = 0.01 
        blocks_to_move = []
        ppos, npos = torch.tensor(np.array(prev_state)), torch.tensor(np.array(next_state))

        for i in range(ppos.shape[0]):
            diff = torch.sum(torch.abs(ppos[i]  - npos[i]), dim = -1)
            # check if there is > 1 block to move; these are mistakes
            if diff[diff> TOL].shape[0] > 1:
                # sentinel to skip later 
                block_to_move = -1
            else:
                block_to_move = torch.argmax(diff).item() + 1

            blocks_to_move.append(block_to_move) 

        return torch.tensor(blocks_to_move, dtype=torch.long).reshape(1,1) 

    def make_3d_positions(self, positions, make_z = False, batch_size = 1, do_infilling = True):
        """
        take (x,y,z) positions and turn them into 1-hot 
        vector over block ids in a x,y,z coordinate grid 
        """
        # positions: 1 for each block id 
        def absolute_to_relative(coord, dim): 
            # shift, now out of 2 
            coord += 1
            # get perc 
            coord = coord / 2
            # scale 
            #real_dim = dim/2
            scaled = coord * dim 
            # shift 
            #scaled += real_dim 
            return int(np.around(scaled))

        image_positions = []
        # create a grid d x w x h
        if make_z: 
            height, width, depth = 7, self.resolution, self.resolution
            for i, position_list in enumerate(positions): 
                image = np.zeros((width, depth, height, 1)) 

                for block_idx, (x, y, z) in enumerate(position_list): 
                    new_x,  new_z = (absolute_to_relative(x, width),
                                          absolute_to_relative(z, depth) )
                    
                    y_val = int(7 * 1 * y) 
                    # infilling 
                    if do_infilling: 
                        offset = int(self.block_size/2)
                        for x_val in range(new_x - offset, new_x + offset):
                            for z_val in range(new_z - offset, new_z + offset):
                                try:
                                    image[x_val, z_val, y_val] = block_idx + 1
                                except IndexError:
                                    # at the edges 
                                    pass 
                    else:
                        image[new_x, new_z, y_val] = block_idx + 1 
                        #image[new_z, new_x, y_val] = block_idx + 1 

                image  = torch.tensor(image).float() 
                image = image.unsqueeze(0)
                # batch, n_labels,  width, height, depth 
                # tile for loss 
                if batch_size > 0:
                    image = torch.cat([image.clone() for i in range(batch_size)], dim = 0)
                image_positions.append(image) 

        else:
            # TODO (elias) change input so it shows the top-most element 
            depth, width = self.resolution, self.resolution
            for i, position_list in enumerate(positions): 
                image = np.zeros(( width, depth, 1 + 1)) 
                for block_idx, (x, y, z) in enumerate(position_list): 
                    new_x, new_z = (absolute_to_relative(x, width),
                                          absolute_to_relative(z, depth)) 
                    offset = 2
                    # infilling 
                    for x_val in range(new_x - offset, new_x + offset):
                        for z_val in range(new_z - offset, new_z + offset):
                            image[x_val, z_val, 0] = block_idx + 1

                            # can only have 7 vertical positions so mod it 
                            image[x_val, z_val, 1] = y % 7

                image  = torch.tensor(image).float() 
                image = image.unsqueeze(0)
                # empty, batch, n_labels, depth, width, height 
                image = image.permute(0, 3, 1, 2) 
                image_positions.append(image) 

        return image_positions

class SimpleTrajectory(BaseTrajectory):
    def __init__(self,
                 line_id: int,
                 commands: List[str], 
                 previous_positions: Tuple[float],
                 previous_rotations: Tuple[float],
                 next_positions: List[Tuple[float]], 
                 next_rotations: List[Tuple[float]], 
                 images: List[str],
                 lengths: List[int],
                 tokenizer: Tokenizer,
                 traj_type: str,
                 batch_size: int,
                 do_filter: bool,
                 do_one_hot: bool, 
                 resolution: int, 
                 top_only: bool,
                 binarize_blocks: bool,
                 image_path: str = None):
        super(SimpleTrajectory, self).__init__(line_id=line_id,
                                               commands=commands, 
                                               previous_positions=previous_positions,
                                               previous_rotations=previous_rotations,
                                               next_positions=next_positions, 
                                               next_rotations=next_rotations, 
                                               images=images,
                                               lengths=lengths,
                                               traj_type=traj_type,
                                               batch_size=batch_size,
                                               resolution=resolution, 
                                               do_filter=do_filter,
                                               do_one_hot = do_one_hot,
                                               top_only=top_only,
                                               binarize_blocks=binarize_blocks,
                                               image_path = image_path) 
        self.tokenizer = tokenizer
        # commands is a list of #timestep text strings 
        self.commands = self.tokenize(commands)
        self.image_path = image_path
        if self.image_path is not None:
            self.image_path = pathlib.Path(self.image_path)

    def tokenize(self, command): 
        # lowercase everything 
        command = [str(x).lower() for x in self.tokenizer(command)]
        self.lengths = [len(command)]
        # add to vocab 
        self.traj_vocab |= set(command) 
        return command

class ImageTrajectory(SimpleTrajectory):
    def __init__(self, *args, **kwargs):
        super(ImageTrajectory, self).__init__(*args, **kwargs) 

    def get_input_positions(self):            
        def imread_safe(path):
            try:
                image = cv2.imread(str(path))
                print(path)
                image = cv2.resize(image, (self.resolution,self.resolution), interpolation = cv2.INTER_AREA)
                # add border to right edge  
                path = pathlib.Path(path)  
                #cv2.imwrite(f"/home/estengel/scratch/debug_images/{path.stem}.png", image)
                #pdb.set_trace() 
                return image.astype(float)
            except SystemError:
                raise SystemError(f"CV2 couldn't read image at {path}, may be invalid path")

        # get image names 
        color_image_path = self.image_path.joinpath("color-heightmaps")
        color_image_names = [color_image_path.joinpath(x) for x in self.images]
        depth_image_path = self.image_path.joinpath("depth-heightmaps")
        depth_image_names = [depth_image_path.joinpath(x) for x in self.images]
        # read images 
        color_images = [imread_safe(x) for x in color_image_names]
        depth_images = [imread_safe(x) for x in depth_image_names]
        # tensorize and concatenate 
        color_images = [torch.tensor(x).permute(2,0,1).float() for x in color_images]
        depth_images = [torch.tensor(x).permute(2,0,1).float() for x in depth_images]
        combined_images = [torch.cat([x,y], dim=0).unsqueeze(0) for (x, y) in zip(color_images, depth_images)]

        return combined_images 

class DatasetReader:
    def __init__(self, train_path,
                       dev_path,
                       test_path,
                       image_path = None,
                       batch_by_line=False,
                       traj_type = "flat",
                       batch_size = 32,
                       max_seq_length = 65,
                       do_filter: bool = False,
                       do_one_hot: bool = False,
                       resolution: int = 64, 
                       top_only: bool = True, 
                       is_bert: bool = False,
                       binarize_blocks: bool = False): 
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.batch_by_line = batch_by_line
        self.traj_type = traj_type
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.do_filter = do_filter
        self.do_one_hot = do_one_hot
        self.top_only = top_only 
        self.binarize_blocks = binarize_blocks
        self.resolution = resolution
        self.is_bert = is_bert
        self.image_path = image_path 
        if self.image_path is None:
            self.trajectory_class = SimpleTrajectory
        else:
            self.trajectory_class = ImageTrajectory
        

        nlp = English()
        self.tokenizer = Tokenizer(nlp.vocab)

        self.paths = {"train": self.train_path,
                      "dev": self.dev_path,
                      "test": self.test_path} 

        self.data = {"train": [], 
                     "dev": [],
                     "test": []} 

    def read_data(self, split) -> set:
        """
        extract trajectories for training and evaluation 
        """
        if split not in self.paths:
            raise AssertionError(f"split {split} not valid. Options are {self.path.keys()}") 
        vocab = set() 
        with open(self.paths[split]) as f1:
            for line_id, line in enumerate(f1.readlines()): 
                line_data = json.loads(line) 
                side_length = line_data["side_length"]

                images = line_data["images"]
                positions = line_data["states"]
                rotations = line_data["rotations"]
                commands = sorted(line_data["notes"], key = lambda x: x["start"])
                # split off previous and subsequent positions and rotations 
                previous_positions, next_positions = positions[0:-1], positions[1:]
                previous_rotations, next_rotations = rotations[0:-1], rotations[1:]

                for timestep in range(len(previous_positions)):
                    command_group = commands[timestep]
                    #for i, command_group in enumerate(commands):
                    for command in command_group["notes"]:  
                        # TODO: add rotations later? 
                        trajectory = self.trajectory_class(line_id,
                                                    command, 
                                                    [previous_positions[timestep]],
                                                    [None],
                                                    [next_positions[timestep]],
                                                    [None],
                                                    images=[images[timestep],images[timestep+1]],
                                                    lengths = [None],
                                                    tokenizer=self.tokenizer,
                                                    traj_type=self.traj_type,
                                                    batch_size=1,
                                                    do_filter=self.do_filter,
                                                    do_one_hot = self.do_one_hot, 
                                                    resolution=self.resolution, 
                                                    top_only = self.top_only,
                                                    binarize_blocks = self.binarize_blocks,
                                                    image_path = self.image_path) 
                        self.data[split].append(trajectory) 
                        vocab |= trajectory.traj_vocab

        if not self.batch_by_line:
            # shuffle and batch data 
            self.shuffle_and_batch_trajectories(split)

        return vocab

    def batchify(self, batch_as_list): 
        """
        pad and tensorize 
        """

        commands = []
        prev_pos_input = []
        prev_pos_for_pred = []
        prev_pos_for_acc  = []
        next_pos_for_pred = []
        next_pos_for_acc = []
        next_pos_for_regression = []
        block_to_move = []
        image = []
        # get max len 
        if not self.is_bert:
            max_length = min(self.max_seq_length, max([traj.lengths[0] for traj in batch_as_list])) 
        else:
            max_length = self.max_seq_length 

        length = []
        for idx in range(len(batch_as_list)):
            traj = batch_as_list[idx]
            # trim! 
            if len(traj.commands) > max_length:
                traj.commands = traj.commands[0:max_length]

            # sentinel value for when multiple blocks moved in a single turn 
            if traj.blocks_to_move[0].item() == -1:
                command = " ".join(traj.commands)
                print(f"SKIPPING {command} for having multiple blocks")
                continue

            length.append(len(traj.commands))
            commands.append(traj.commands + [PAD for i in range(max_length - len(traj.commands))])
            prev_pos_input.append(traj.previous_positions_input[0])
            prev_pos_for_pred.append(traj.previous_positions_for_pred[0]) 
            prev_pos_for_acc.append(traj.previous_positions_for_acc[0]) 

            next_pos_for_pred.append(traj.next_positions_for_pred[0])
            next_pos_for_acc.append(traj.next_positions_for_acc[0])

            block_to_move.append(traj.blocks_to_move[0].long())
            try:
                btmv = block_to_move[-1] - 1
                next_pos_for_regression.append(torch.tensor(traj.next_positions_for_regression[0][btmv]))
            except IndexError:
                pdb.set_trace() 
            image.append(traj.images)  
        
        if len(prev_pos_input) == 0:
            return None 

        prev_pos_input = torch.cat(prev_pos_input, 0)
        prev_pos_for_pred = torch.cat(prev_pos_for_pred, 0)
        prev_pos_for_acc  = torch.cat(prev_pos_for_acc, 0)
        next_pos_for_pred  = torch.cat(next_pos_for_pred, 0) 
        next_pos_for_acc = torch.cat(next_pos_for_acc, 0) 
        next_pos_for_regression = torch.cat(next_pos_for_regression, 0) 
        block_to_move = torch.cat(block_to_move, 0) 

        return {"command": commands,
                "prev_pos_input": prev_pos_input,
                "prev_pos_for_acc": prev_pos_for_acc,
                "prev_pos_for_pred": prev_pos_for_pred,
                "next_pos_for_acc": next_pos_for_acc,
                "next_pos_for_pred": next_pos_for_pred,
                "next_pos_for_regression": next_pos_for_regression,
                "block_to_move": block_to_move,
                "image": image,
                "length": length} 

    def shuffle_and_batch_trajectories(self, split=None): 
        """
        shuffle the trajectories and batch them together 
        """
        if split is None:
            keys = self.data.keys()
        else:
            keys = [split]
        for key in keys:
            split_data = self.data[key]
            # shuffle data 
            np.random.shuffle(split_data)
            # batchify
            batches = []
            curr_batch = []
            for i in range(len(split_data)):
                curr_batch.append(split_data[i]) 
                if (i + 1) % self.batch_size == 0:
                    batch = self.batchify(curr_batch)
                    if batch is not None:
                        batches.append(batch) 
                    curr_batch = []
            # append last one 
            if len(curr_batch) > 0: 
                batch = self.batchify(curr_batch)
                if batch is not None: 
                    batches.append(batch) 

            self.data[key] = batches  

class GoodRobotDatasetReader: 
    def __init__(self, 
                path_or_obj: str,
                split_type: str = 'random',
                task_type: str = "rows-and-stacks",
                color_pair: List[str] = None, 
                augment_by_flipping: bool = True, 
                augment_by_rotating: bool = True, 
                augment_with_noise: bool = False, 
                augment_language: bool = True, 
                noise_num_samples: int = 2,
                leave_out_color: str = None, 
                batch_size: int = 32,
                max_seq_length: int = 60,
                resolution: int = 64,
                is_bert: bool = True,
                overfit: bool = False):
        # TODO(elias) add depth heightmaps 
        self.batch_size = batch_size
        self.is_bert = is_bert 
        self.resolution = resolution 
        self.max_seq_length = max_seq_length

        self.noise_gaussian_params = [0.0, 0.05]
        self.noise_num_samples = noise_num_samples


        if type(path_or_obj) == str:
            self.path = pathlib.Path(path_or_obj)
            self.pkl_files = self.path.glob("*/*.pkl")
            self.all_data = []
            for pkl_file in self.pkl_files:
                with open(pkl_file, "rb") as f1:
                    data = pkl.load(f1)
                    self.all_data.extend(data) 
        else:
            self.all_data = [path_or_obj]

        if task_type == "rows":
            self.all_data = GoodRobotDatasetReader.filter_data(self.all_data, rows=True) 
        elif task_type == "stacks": 
            self.all_data = GoodRobotDatasetReader.filter_data(self.all_data, rows=False) 

        if split_type == "random": 
            np.random.shuffle(self.all_data)
            train_len = int(0.8 * len(self.all_data))
            devtest_len = int(0.1 * len(self.all_data))
            train_data = self.all_data[0:train_len]
            dev_data = self.all_data[train_len: train_len + devtest_len]
            test_data = self.all_data[train_len + devtest_len: ]
        elif split_type == "none": 
            train_data = self.all_data
            dev_data = self.all_data
            test_data = self.all_data 
        elif split_type == "leave-out-color":
            # train on everything except (<color_a>, <color_b>) combos in either direction
            allowed_data = [x for x in self.all_data if not(x.source_code in color_pair and x.target_code in color_pair)]
            held_out_data = [x for x in self.all_data if x.source_code in color_pair and x.target_code in color_pair]
            train_data = allowed_data
            held_out_len = len(held_out_data)
            dev_len = int(held_out_len/3)
            dev_data = held_out_data[0:dev_len]
            test_data = held_out_data[dev_len: ]

        elif split_type == "train-stack-test-row":
            pass
        elif split_type == "train-row-test-stack":
            pass
        else:
            raise AssertionError(f"split strategy {split_type} is invalid")
        
        # only augment train data 
        # augment by flipping across 4 axes 
        if augment_by_flipping:
            new_data = []
            for pair in train_data:
                for axis in range(1,5):
                    new_pair = flip_pair(pair, axis) 
                    new_data.append(new_pair) 
            train_data += new_data 

        if augment_by_rotating:
            new_data = []
            for pair in train_data:
                for rot in range(1, 4):
                    new_pair = rotate_pair(pair, rot)
                    new_data.append(new_pair)
            train_data += new_data

        if augment_with_noise:
            new_data = []
            print(f"augmenting with noise")
            for pair in tqdm(train_data):
                for i in range(self.noise_num_samples):
                    new_pair = gaussian_augment(pair, self.noise_gaussian_params)
                    new_data.append(new_pair)
            train_data += new_data
            print(f"added")


        self.color_names = ['blue', 'green', 'yellow', 'red', 'brown', 'orange', 'gray', 'purple', 'cyan', 'pink']

        self.data = {"train": train_data,
                     "dev": dev_data,
                     "test": test_data}

        print(f"DATA STATS: train: {len(train_data)}, dev: {len(dev_data)}, test: {len(test_data)}")

        if overfit:
            # for debugging, overfit just to dev 
            response = input( f"WARNING: OVERFITTING TO DEV DATA. CONTINUE? (y/n)")
            if response == 'y':
                pass
            else:
                print("EXITING...")
                sys.exit() 
            self.data = {"train": dev_data[3:6],
                         "dev": dev_data[3:6],
                         "test": dev_data[3:6]} 

        self.vocab = set()
        for pair in self.data['train']:
            if 'bad' in [pair.source_code, pair.target_code]:
                continue
            command = pair.generate()
            self.vocab |= set(re.split("\s+", command))

        self.shuffle_and_batch_trajectories()
        
    @staticmethod
    def leave_out_color(data, color_pair):
        # run through all data, make splits so that one color pair is only seen at test time 
        pass 

    @staticmethod
    def filter_data(data, rows = False):
        filtered_data = []
        for pair in data:
            # rows 
            if rows and pair.is_row:
                filtered_data.append(pair)
            if not rows and (not hasattr(pair, "is_row") or not pair.is_row): 
                filtered_data.append(pair)
        return filtered_data 

    def shuffle_and_batch_trajectories(self, split=None): 
        """
        shuffle the trajectories and batch them together 
        """
        if split is None:
            keys = self.data.keys()
        else:
            keys = [split]
        for key in keys:
            split_data = self.data[key]
            # shuffle data 
            np.random.shuffle(split_data)
            # batchify
            batches = []
            curr_batch = []
            for i in range(len(split_data)):
                curr_batch.append(split_data[i]) 
                if (i + 1) % self.batch_size == 0:
                    batch = self.batchify(curr_batch)
                    if batch is not None:
                        batches.append(batch) 
                    curr_batch = []
            # append last one 
            if len(curr_batch) > 0: 
                batch = self.batchify(curr_batch)
                if batch is not None: 
                    batches.append(batch) 

            self.data[key] = batches  

    def batchify(self, batch_as_list): 
        """
        pad and tensorize 
        """

        commands = []
        prev_pos_input = []
        prev_pos_for_pred = []
        prev_pos_for_acc  = []
        prev_pos_for_vis = []
        next_pos_for_pred = []
        next_pos_for_acc = []
        next_pos_for_vis = []
        next_pos_for_regression = []
        block_to_move = []

        image = []
        pairs = []
        # get max len 
        if not self.is_bert:
            max_length = min(self.max_seq_length, max([len(re.split("\s+", pair.generate())) for pair in batch_as_list]))
        else:
            max_length = self.max_seq_length 

        length = []
        for idx in range(len(batch_as_list)):
            pair = copy.deepcopy(batch_as_list[idx])
            # trim! 
            command = re.split("\s+", pair.generate()) 
            if len(command) > max_length:
                command = command[0:max_length]

            if 'bad' in [pair.source_code, pair.target_code]:
                continue
            
            pair.resolution = self.resolution 
            pair.resize()

            length.append(len(command))
            commands.append(command + [PAD for i in range(max_length - len(command))])

            prev_pos_input.append(torch.from_numpy(pair.prev_image.copy()).unsqueeze(0))
            if pair.prev_location is not None: 
                prev_pos_for_pred.append(torch.from_numpy(pair.get_mask(pair.prev_location).copy()).unsqueeze(0))
                prev_pos_for_vis.append(torch.from_numpy(pair.prev_image.copy()).unsqueeze(0))
                prev_pos_for_acc.append(torch.from_numpy(pair.prev_state_image.copy()).unsqueeze(0))

            if pair.next_location is not None:
                next_pos_for_pred.append(torch.from_numpy(pair.get_mask(pair.next_location).copy()).unsqueeze(0))
                next_pos_for_vis.append(torch.from_numpy(pair.next_image.copy()).unsqueeze(0))
            pairs.append(pair)
            block_to_move.append(self.color_names.index(pair.source_code))

        prev_pos_input = torch.cat(prev_pos_input, 0)
        if len(prev_pos_for_pred) > 0:
            prev_pos_for_pred = torch.cat(prev_pos_for_pred, 0)
            prev_pos_for_pred = prev_pos_for_pred.float().unsqueeze(-1)
        if len(prev_pos_for_acc) > 0:
            prev_pos_for_acc  = torch.cat(prev_pos_for_acc, 0)
            prev_pos_for_acc  = prev_pos_for_acc.float() 
        if len(prev_pos_for_vis) > 0:
            prev_pos_for_vis  = torch.cat(prev_pos_for_vis, 0)
            prev_pos_for_vis  = prev_pos_for_vis.float()
        if len(next_pos_for_pred) > 0:
            next_pos_for_pred  = torch.cat(next_pos_for_pred, 0) 
            next_pos_for_pred = next_pos_for_pred.float().unsqueeze(-1)
        if len(next_pos_for_acc) > 0:
            next_pos_for_acc = torch.cat(next_pos_for_acc, 0) 
            next_pos_for_acc = next_pos_for_acc.float() 

        prev_pos_input = prev_pos_input.permute(0, 3, 1, 2).float() 

        block_to_move = torch.tensor(block_to_move)

        return {"command": commands,
                "prev_pos_input": prev_pos_input,
                "prev_pos_for_acc": prev_pos_for_acc,
                "prev_pos_for_pred": prev_pos_for_pred,
                "prev_pos_for_vis": prev_pos_for_vis,
                "next_pos_for_acc": next_pos_for_acc,
                "next_pos_for_vis": next_pos_for_vis,
                "next_pos_for_pred": next_pos_for_pred,
                "next_pos_for_regression": None,
                "block_to_move":  block_to_move,
                "pairs": pairs, 
                "length": length} 

if __name__ == "__main__":
    reader = DatasetReader("blocks_data/devset.json", "blocks_data/devset.json", "blocks_data/devset.json")  
    reader.read_data("train") 
    #reader = DatasetReader("blocks_data/devset.json", "blocks_data/devset.json", "blocks_data/devset.json", batch_by_line=True)
    #reader.read_data("train") 
    
    for trajectory in reader.data["train"]:
        for instance in trajectory: 
            print(instance)   
            sys.exit() 
