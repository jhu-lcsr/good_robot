import os 
import sys
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from spacy.tokenizer import Tokenizer 
from spacy.lang.en import English
import torch
from torch.nn import functional as F
import pdb 
import copy 
np.random.seed(12) 

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
                 top_only: bool = True, 
                 binarize_blocks: bool = False):

        if batch_size is None:
            batch_size = len(commands)
        self.batch_size = batch_size 
        self.do_filter = do_filter
        self.top_only = top_only
        self.binarize_blocks = binarize_blocks
        self.resolution = resolution

        self.line_id = line_id
        self.commands = commands

        self.blocks_to_move = self.get_blocks_to_move(previous_positions, next_positions).float() 

        # output for previous positions has depth that gets filtered 
        self.previous_positions_for_pred = self.make_3d_positions(previous_positions, make_z = True, batch_size = self.batch_size)
        self.previous_positions_for_acc = copy.deepcopy(self.previous_positions_for_pred) 
        self.next_positions_for_pred = self.make_3d_positions(next_positions, make_z = True, batch_size = self.batch_size)
        self.next_positions_for_acc = copy.deepcopy(self.next_positions_for_pred) 
        #self.blocks_to_move = self.get_blocks_to_move()

        self.previous_positions = previous_positions
        self.next_positions = next_positions
        self.previous_rotations = previous_rotations
        self.next_rotations = next_rotations

        if self.top_only:
            (self.previous_positions_for_pred, 
            self.previous_positions_for_acc,
            self.next_positions_for_pred, 
            self.next_positions_for_acc) = self.filter_top_only(self.previous_positions_for_pred, 
                                                                self.previous_positions_for_acc, 
                                                                self.next_positions_for_pred, 
                                                                self.next_positions_for_acc)

        # set as input but one-hot 
        self.previous_positions_input = self.one_hot(copy.deepcopy(self.previous_positions_for_acc)) 

        if self.do_filter: 
            # for loss, only look at the single block moved 
            self.previous_positions_for_pred = self.filter_by_blocks_to_move(self.previous_positions_for_pred)
            self.next_positions_for_pred = self.filter_by_blocks_to_move(self.next_positions_for_pred)

        self.images = images
        self.lengths = lengths
        self.traj_vocab = set() 
        self.traj_type = traj_type

    #def one_hot_helper(self, input_data, C=21):
    #    # states_before_onehot.shape = T,64,64
    #    input_data = input_data.reshape(-1, 64, 64)
    #    states_before_onehot = input_data.unsqueeze_(1).long()  # convert to Tx1xHxW
    #    one_hot = torch.FloatTensor(states_before_onehot.size(0),C, states_before_onehot.size(2), states_before_onehot.size(3)).zero_()
    #    one_hot.scatter_(1, states_before_onehot, 1)
    #    return one_hot

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

        TOL = 0.001 
        blocks_to_move = []
        ppos, npos = torch.tensor(np.array(prev_state)), torch.tensor(np.array(next_state))

        for i in range(ppos.shape[0]):
            diff = torch.sum(torch.abs(ppos[i]  - npos[i]), dim = -1)
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
                        offset = 2
                        for x_val in range(new_x - offset, new_x + offset):
                            for z_val in range(new_z - offset, new_z + offset):
                                try:
                                    image[x_val, z_val, y_val] = block_idx + 1
                                except IndexError:
                                    # at the edges 
                                    pass 
                    else:
                        image[new_x, new_z, y_val] = block_idx + 1 

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
                 resolution: int, 
                 top_only: bool,
                 binarize_blocks: bool):
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
                                               top_only=top_only,
                                               binarize_blocks=binarize_blocks) 
        self.tokenizer = tokenizer
        # commands is a list of #timestep text strings 
        self.commands = self.tokenize(commands)

    def tokenize(self, command): 
        # lowercase everything 
        command = [str(x).lower() for x in self.tokenizer(command)]
        self.lengths = [len(command)]
        # add to vocab 
        self.traj_vocab |= set(command) 
        return command


class DatasetReader:
    def __init__(self, train_path,
                       dev_path,
                       test_path,
                       batch_by_line=False,
                       traj_type = "flat",
                       batch_size = 32,
                       max_seq_length = 65,
                       do_filter: bool = False,
                       resolution: int = 64, 
                       top_only: bool = True, 
                       binarize_blocks: bool = False): 
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.batch_by_line = batch_by_line
        self.traj_type = traj_type
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.do_filter = do_filter
        self.top_only = top_only 
        self.binarize_blocks = binarize_blocks
        self.resolution = resolution

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
                        trajectory = SimpleTrajectory(line_id,
                                                    command, 
                                                    [previous_positions[timestep]],
                                                    [previous_rotations[timestep]],
                                                    [next_positions[timestep]],
                                                    [next_rotations[timestep]],
                                                    images=[images[timestep],images[timestep+1]],
                                                    lengths = [None],
                                                    tokenizer=self.tokenizer,
                                                    traj_type=self.traj_type,
                                                    batch_size=1,
                                                    do_filter=self.do_filter,
                                                    resolution=self.resolution, 
                                                    top_only = self.top_only,
                                                    binarize_blocks = self.binarize_blocks) 
                        self.data[split].append(trajectory) 
                        vocab |= trajectory.traj_vocab

        if not self.batch_by_line:
            # shuffle and batch data 
            self.shuffle_and_batch_trajectories(split)

        return vocab

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
                    batches.append(self.batchify(curr_batch))
                    curr_batch = []
            # append last one 
            if len(curr_batch) > 0: 
                batches.append(self.batchify(curr_batch) )

            self.data[key] = batches  

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
        block_to_move = []
        image = []
        # get max len 
        max_length = min(self.max_seq_length, max([traj.lengths[0] for traj in batch_as_list])) 

        # sort lengths
        #lengths = [] 
        #for i, traj in enumerate(batch_as_list):
        #    com_len = len(traj.commands)
        #    lengths.append((i, com_len))
        #sorted_lengths = sorted(lengths, key = lambda x: x[1], reverse=True) 
        #idxs, length = zip(*sorted_lengths) 
        #length = list(length) 
        # iterate sorted by length 
        #for i, (idx, l) in enumerate(sorted_lengths): 
        length = []
        for idx in range(len(batch_as_list)):
            traj = batch_as_list[idx]
            # trim! 
            if len(traj.commands) > max_length:
                traj.commands = traj.commands[0:max_length]

            length.append(len(traj.commands))
            commands.append(traj.commands + [PAD for i in range(max_length - len(traj.commands))])
            prev_pos_input.append(traj.previous_positions_input[0])
            prev_pos_for_pred.append(traj.previous_positions_for_pred[0]) 
            prev_pos_for_acc.append(traj.previous_positions_for_acc[0]) 

            next_pos_for_pred.append(traj.next_positions_for_pred[0])
            next_pos_for_acc.append(traj.next_positions_for_acc[0])

            block_to_move.append(traj.blocks_to_move[0].long())
            image.append(traj.images)  
        

        prev_pos_input = torch.cat(prev_pos_input, 0)
        prev_pos_for_pred = torch.cat(prev_pos_for_pred, 0)
        prev_pos_for_acc  = torch.cat(prev_pos_for_acc, 0)
        next_pos_for_pred  = torch.cat(next_pos_for_pred, 0) 
        next_pos_for_acc = torch.cat(next_pos_for_acc, 0) 
        block_to_move = torch.cat(block_to_move, 0) 

        return {"command": commands,
                "prev_pos_input": prev_pos_input,
                "prev_pos_for_acc": prev_pos_for_acc,
                "prev_pos_for_pred": prev_pos_for_pred,
                "next_pos_for_acc": next_pos_for_acc,
                "next_pos_for_pred": next_pos_for_pred,
                "block_to_move": block_to_move,
                "image": image,
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
