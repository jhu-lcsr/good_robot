import os 
import sys
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from spacy.tokenizer import Tokenizer 
from spacy.lang.en import English
import torch
np.random.seed(12) 

PAD = "<PAD>"

class FlatIterator:
    """
    shuffle trajectories and give pairs, allowing for greater batching 
    """
    def __init__(self, 
                all_trajectories, 
                batch_size = 32):
        self.all_trajectories = all_trajectories
        self._index = -1
        self.all_timesteps = None
        self.batch_size = batch_size

    def __next__(self):
        if self._index + 1 < len(self.all_timesteps):
            self._index += 1
            try:
                command, prev_pos, prev_pos_for_acc, next_pos, prev_rot, next_rot, block_to_move, image, length = self.all_timesteps[self._index]
            except ValueError:
                command, prev_pos, prev_pos_for_acc, next_pos, prev_rot, next_rot, block_to_move, length = self.all_timesteps[self._index]
                image=None

            return {"command": command,
                    "previous_position": prev_pos,
                    "next_position": next_pos,
                    "previous_position_for_acc": prev_pos_for_acc,
                    "previous_rotation": prev_rot,
                    "next_rotation": next_rot,
                    "block_to_move": block_to_move, 
                    "image": image,
                    "length": length} 

        else: 
            raise StopIteration 


class TrajectoryIterator:
    """
    Iterator over trajectory that returns a single timestep from the Trajectory
    meaning: 1 command, 1 previous position, 1 next position. Commands may be 
    batched, but positions are not. 
    """
    def __init__(self, traj):
        self.traj = traj
        self._index = -1

    def __next__(self):
        if self._index + 1 < len(self.traj.to_iterate):
            self._index += 1
            try:
                command, prev_pos, prev_pos_for_acc, next_pos, prev_rot, next_rot, block_to_move, image, length = self.traj.to_iterate[self._index]
            except ValueError:
                command, prev_pos, prev_pos_for_acc, next_pos, prev_rot, next_rot, block_to_move, length = self.traj.to_iterate[self._index]
                image=None
           
            return {"command": command,
                    "previous_position": prev_pos,
                    "next_position": next_pos,
                    "previous_position_for_acc": prev_pos_for_acc,
                    "previous_rotation": prev_rot,
                    "next_rotation": next_rot,
                    "block_to_move": block_to_move, 
                    "image": image,
                    "length": length} 

        else: 
            raise StopIteration 

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
                 batch_size: int = None):

        if batch_size is None:
            batch_size = len(commands)
        self.batch_size = batch_size 

        self.line_id = line_id
        self.commands = commands

        self.previous_positions = self.make_3d_positions(previous_positions, batch_size = len(commands))
        self.previous_positions_for_acc = self.make_3d_positions(previous_positions, make_z = True, batch_size = self.batch_size)
        self.previous_rotations = previous_rotations
        self.next_positions = self.make_3d_positions(next_positions, make_z = True, batch_size = self.batch_size)
        self.next_rotations = next_rotations
        self.blocks_to_move = self.get_blocks_to_move()

        self.images = images
        self.lengths = lengths
        self.traj_vocab = set() 
        self.traj_type = traj_type

        self.to_iterate = list(zip(self.commands, 
                                   self.previous_positions, 
                                   self.previous_positions_for_acc,
                                   self.next_positions, 
                                   self.previous_rotations, 
                                   self.next_rotations, 
                                   self.blocks_to_move,
                                   self.images, 
                                   self.lengths)) 

        # filter out empty commands
        self.to_iterate = [x for x in self.to_iterate if len(x[0]) > 0 ]


    def __iter__(self):
        if self.traj_type == "flat":     
            return FlatIterator(self)
        else:
            return TrajectoryIterator(self)

    def get_blocks_to_move(self):
        batch_idx = 0
        bsz = self.previous_positions_for_acc[0].shape[0]
        bad = 0
        all_blocks_to_move = []
        for timestep in range(len(self.previous_positions_for_acc)):
            prev_pos = self.previous_positions_for_acc[timestep][batch_idx] 
            next_pos = self.next_positions[timestep][batch_idx]
            different_pixels = prev_pos[prev_pos != next_pos]
            # exclude background 
            different_pixel_idx = different_pixels[different_pixels != 0]
            try:
                blocks_to_move = torch.ones((bsz, 1), dtype=torch.int64) * different_pixel_idx.item() 
            except ValueError:
                try:
                    blocks_to_move = torch.ones((bsz, 1), dtype=torch.int64) * different_pixel_idx[0].item() 
                except IndexError:
                    blocks_to_move = torch.zeros((bsz, 1), dtype=torch.int64) 
                bad += 1
            all_blocks_to_move.append(blocks_to_move) 
        #print(f"there are {bad} blocks with >1 move") 
        return all_blocks_to_move

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
            height, width, depth = 4, 64, 64
            for i, position_list in enumerate(positions): 
                image = np.zeros((width, depth, height, 1)) 

                for block_idx, (x, y, z) in enumerate(position_list): 
                    new_x,  new_z = (absolute_to_relative(x, width),
                                          absolute_to_relative(z, depth) )
                    
                    y_val = int(4 * 1 * y) 
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
            depth, width = 64, 64 

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

                            # can only have 4 vertical positions so mod it 
                            image[x_val, z_val, 1] = y % 4

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
                 batch_size: int):
        super(SimpleTrajectory, self).__init__(line_id,
                                         commands, 
                                         previous_positions,
                                         previous_rotations,
                                         next_positions, 
                                         next_rotations, 
                                         images,
                                         lengths,
                                         traj_type,
                                         batch_size) 
        self.tokenizer = tokenizer
        # commands is a list of #timestep text strings 
        self.commands = self.tokenize(commands)

    def tokenize(self, command): 
        command = [str(x) for x in self.tokenizer(command)]
        self.lengths = [len(command)]
        # add to vocab 
        self.traj_vocab |= set(command) 
        return command

class BatchedTrajectory(BaseTrajectory): 
    """
    batches trajectories so that all 9 annotator commands with shared 
    positions and rotations are batched together. 
    """
    def __init__(self,
                 line_id: int,
                 commands: List[List[str]], 
                 previous_positions: Tuple[float],
                 previous_rotations: Tuple[float],
                 next_positions: List[Tuple[float]], 
                 next_rotations: List[Tuple[float]], 
                 images: List[str],
                 lengths: List[str],
                 tokenizer: Tokenizer,
                 traj_type: str):  
        super(BatchedTrajectory, self).__init__(line_id,
                                                commands,
                                                previous_positions,
                                                previous_rotations,
                                                next_positions,
                                                next_rotations,
                                                images,
                                                lengths,
                                                traj_type)

        # commands is now a 9xlen matrix 
        self.tokenizer = tokenizer
        self.commands, self.lengths = self.pad_commands(self.tokenize(commands)) 
        self.traj_vocab = set() 

        # override 
        self.to_iterate = list(zip(self.commands, 
                                   self.previous_positions, 
                                   self.previous_positions_for_acc,
                                   self.next_positions, 
                                   self.previous_rotations, 
                                   self.next_rotations, 
                                   self.blocks_to_move,
                                   self.images, 
                                   self.lengths)) 


    def tokenize(self, commands): 
        for annotator_idx, command_list in enumerate(commands):
            for timestep, command in enumerate(command_list):
                commands[annotator_idx][timestep] = list(self.tokenizer(command))
                # add to vocab 
                self.traj_vocab |= set(commands[annotator_idx][timestep])
        return commands 

    def pad_commands(self, commands): 
        # pad commands in a batch to have the same length
        max_len = 0
        lengths = [[max_len for j in range(len(commands))] for i in range(len(commands[0]))]
        pad_str = [PAD for __ in range(max_len)]
        new_commands = [[pad_str for j in range(len(commands))] for i in range(len(commands[0]))]
        # get max length 
        for annotator_idx, command_list in enumerate(commands):
            for command_seq in command_list:
                if len(command_seq) > max_len:
                    max_len = len(command_seq) 

        # pad to max length 
        for annotator_idx, command_list in enumerate(commands):
            for i, command_seq in enumerate(command_list): 
                lengths[i][annotator_idx] = len(command_seq)
                #commands[annotator_idx][i] += ["<PAD>" for i in range(max_len - len(command_seq))]
                # permute 0 and 1, trajectory timestep first then annotator 
                new_commands[i][annotator_idx] = commands[annotator_idx][i] +\
                                            ["<PAD>" for i in range(max_len - len(command_seq))]

        return new_commands, lengths

class DatasetReader:
    def __init__(self, train_path,
                       dev_path,
                       test_path,
                       batch_by_line=False,
                       traj_type = "flat",
                       batch_size = 32,
                       max_seq_length = 65): 
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.batch_by_line = batch_by_line
        self.traj_type = traj_type
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

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
                commands = line_data["notes"]
                # split off previous and subsequent positions and rotations 
                previous_positions, next_positions = positions[0:-1], positions[1:]
                previous_rotations, next_rotations = rotations[0:-1], rotations[1:]

                #command_trajectories = [[] for i in range(len(commands[0]['notes']))]
                if self.batch_by_line:
                    command_trajectories = [[] for i in range(9)]
                    # split commands out into separate trajectories  
                    for i, step in enumerate(commands):
                        for j, annotation in enumerate(step["notes"]): 
                            # 9 annotations, 3 per annotator 
                            if j > 8:
                                break 
                            command_trajectories[j].append(annotation) 
                    trajectory = BatchedTrajectory(line_id, 
                                                   command_trajectories,
                                                   previous_positions,
                                                   previous_rotations,
                                                   next_positions,
                                                   next_rotations,
                                                   images = images,
                                                   lengths = [0]*len(command_trajectories),
                                                   tokenizer = self.tokenizer,
                                                   traj_type = self.traj_type) 
                    self.data[split].append(trajectory)  
                    vocab |= trajectory.traj_vocab
                else:
                    for i, command_group in enumerate(commands):
                        for command in command_group["notes"]:  
                            for timestep in range(len(previous_positions)):
                                # TODO: add rotations later? 
                                trajectory = SimpleTrajectory(line_id,
                                                            command, 
                                                            [previous_positions[timestep]],
                                                            [None],
                                                            [next_positions[timestep]],
                                                            [None],
                                                            images=[images[timestep]],
                                                            lengths = [None],
                                                            tokenizer=self.tokenizer,
                                                            traj_type=self.traj_type,
                                                            batch_size=1)
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

        commands, prev_pos, next_pos, prev_pos_for_acc,  prev_rot, next_rot, block_to_move, image, length = [], [], [], [], [], [], [], [], []
        # get max len 
        max_length = min(self.max_seq_length, max([traj.lengths[0] for traj in batch_as_list])) 
        # pad  
        for i, traj in enumerate(batch_as_list): 
            # trim! 
            if len(traj.commands) > max_length:
                traj.commands = traj.commands[0:max_length]
            
            length.append(len(traj.commands)) 

            commands.append(traj.commands + [PAD for i in range(max_length - len(traj.commands))])
            prev_pos.append(traj.previous_positions[0])
            prev_pos_for_acc.append(traj.previous_positions_for_acc[0]) 
            prev_rot.append(traj.previous_rotations[0])
            next_pos.append(traj.next_positions[0])
            next_rot.append(traj.next_rotations[0]) 
            block_to_move.append(traj.blocks_to_move[0].long())
            image.append(traj.images)  
        

        prev_pos = torch.cat(prev_pos, 0)
        prev_pos_for_acc = torch.cat(prev_pos_for_acc, 0)
        next_pos = torch.cat(next_pos, 0) 
        block_to_move = torch.cat(block_to_move, 0) 

        #print(f"prev_pos {prev_pos.shape}") 
        #print(f"prev_pos_for_acc  {prev_pos_for_acc.shape}") 
        #print(f"next_pos {next_pos.shape}") 
        #print(f"block to move {block_to_move.shape}") 
        #sys.exit() 

        return {"command": commands,
                "previous_position": prev_pos,
                "next_position": next_pos,
                "previous_position_for_acc": prev_pos_for_acc,
                "previous_rotation": prev_rot,
                "next_rotation": next_rot,
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
