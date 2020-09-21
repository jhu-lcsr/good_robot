import os 
import sys
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from spacy.tokenizer import Tokenizer 
from spacy.lang.en import English
import torch

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
                command, prev_pos, prev_pos_for_acc, next_pos, prev_rot, next_rot, image, length = self.traj.to_iterate[self._index]
            except ValueError:
                command, prev_pos, prev_pos_for_acc, next_pos, prev_rot, next_rot, length = self.traj.to_iterate[self._index]
                image=None
            
            return {"command": command,
                    "previous_position": prev_pos,
                    "next_position": next_pos,
                    "previous_position_for_acc": prev_pos_for_acc,
                    "previous_rotation": prev_rot,
                    "next_rotation": next_rot,
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
                 lengths: List[int]):
        self.line_id = line_id
        self.commands = commands
        self.previous_positions = self.make_3d_positions(previous_positions, batch_size = len(commands))
        self.previous_positions_for_acc = self.make_3d_positions(previous_positions, make_z = True, batch_size = len(commands))
        self.previous_rotations = previous_rotations
        self.next_positions = self.make_3d_positions(next_positions, make_z = True, batch_size = len(commands))
        self.next_positions_for_acc  = self.make_3d_positions(next_positions, batch_size = len(commands))
        self.next_rotations = next_rotations
        self.images = images
        self.lengths = lengths
        self.traj_vocab = set() 

        self.to_iterate = list(zip(self.commands, 
                                   self.previous_positions, 
                                   self.previous_positions_for_acc,
                                   self.next_positions, 
                                   self.previous_rotations, 
                                   self.next_rotations, 
                                   self.images, 
                                   self.lengths)) 

        # filter out empty commands
        self.to_iterate = [x for x in self.to_iterate if len(x[0]) > 0 ]


    def __iter__(self):
        return TrajectoryIterator(self)

    def make_3d_positions(self, positions, make_z = False, batch_size = 9):
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

        # create a grid d x w x h
        if make_z: 
            height, width, depth = 4, 64, 64
            n_blocks = 20
            #image = np.zeros((depth, width, height, 1)) 
            image = np.zeros((width, depth, height, 1)) 
            for i, position_list in enumerate(positions): 
                for block_idx, (x, y, z) in enumerate(position_list): 
                    new_x,  new_z = (absolute_to_relative(x, width),
                                          absolute_to_relative(z, depth) )
                    # side length: 0.1524 = 9.7536/64
                    #width, height = 10, 10 
                    #offset = int(width/2)
                    offset = 1
                    y_val = int(4 * 1 * y) 
                    print(f"trying to add block_idx {block_idx} to {x, z, y}, {new_x, new_z, y_val}") 
                    # infilling 
                    for x_val in range(new_x - offset, new_x + offset):
                        #for y_val in range(new_y - offset, new_y + offset):
                        for z_val in range(new_z - offset, new_z + offset):
                            try:
                                image[x_val, z_val, y_val] = block_idx + 1
                            except IndexError:
                                # at the edges 
                                #print(f"error add block_idx {block_idx}") 
                                pass 
            image  = torch.tensor(image).float() 
            image = image.unsqueeze(0)
            # batch, n_labels,  width, height, depth 
            #image = image.permute(0, 4, 2, 3, 1) 
            # tile for loss 
            image = torch.cat([image.clone() for i in range(batch_size)], dim = 0)

        else:
            height, width = 64, 64 
            n_blocks = 20
            image = np.zeros(( width, height, 1 + 1)) 
            for i, position_list in enumerate(positions): 
                for block_idx, (x, y, z) in enumerate(position_list): 
                    new_x, new_y = (absolute_to_relative(x, width),
                                          absolute_to_relative(y, height)) 
                    # side length: 0.1524 => 9.7536/64
                    width, height = 10, 10 
                    offset = int(width/2)

                    # infilling 
                    for x_val in range(new_x - offset, new_x + offset):
                        for y_val in range(new_y - offset, new_y + offset):
                            image[x_val, y_val, 0] = block_idx
                            image[x_val, y_val, 1] = z

            image  = torch.tensor(image).float() 
            image = image.unsqueeze(0)
            # empty, batch, n_labels, depth, width, height 
            image = image.permute(0, 3, 1, 2) 

        return [image]

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
                 tokenizer: Tokenizer):
        super(SimpleTrajectory, self).__init__(line_id,
                                         commands, 
                                         previous_positions,
                                         previous_rotations,
                                         next_positions, 
                                         next_rotations, 
                                         images,
                                         lengths) 
        self.tokenizer = tokenizer
        # commands is a list of #timestep text strings 
        self.commands = self.tokenize(commands)

    def tokenize(self, commands): 
        for timestep, command in enumerate(commands):
            commands[timestep] = list(self.tokenizer(command))
            # add to vocab 
            self.traj_vocab |= set(commands[timestep])
        return commands 

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
                 tokenizer: Tokenizer):
        super(BatchedTrajectory, self).__init__(line_id,
                                                commands,
                                                previous_positions,
                                                previous_rotations,
                                                next_positions,
                                                next_rotations,
                                                images,
                                                lengths) 
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
        pad_str = ["<PAD>" for __ in range(max_len)]
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
                       batch_by_line=False): 
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.batch_by_line = batch_by_line

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
                # TODO: add images  
                # split off previous and subsequent positions and rotations 
                previous_positions, next_positions = positions[0:-1], positions[1:]
                previous_rotations, next_rotations = rotations[0:-1], rotations[1:]

                #command_trajectories = [[] for i in range(len(commands[0]['notes']))]
                command_trajectories = [[] for i in range(9)]
                # split commands out into separate trajectories  
                for i, step in enumerate(commands):
                    for j, annotation in enumerate(step["notes"]): 
                        # 9 annotations, 3 per annotator 
                        if j > 8:
                            break 
                        command_trajectories[j].append(annotation) 
                if self.batch_by_line:
                    trajectory = BatchedTrajectory(line_id, 
                                                   command_trajectories,
                                                   previous_positions,
                                                   previous_rotations,
                                                   next_positions,
                                                   next_rotations,
                                                   images = images,
                                                   lengths = [0]*len(command_trajectories),
                                                   tokenizer = self.tokenizer) 
                    self.data[split].append(trajectory)  

                    vocab |= trajectory.traj_vocab

                else:
                    for command_traj in command_trajectories:
                        trajectory = SimpleTrajectory(line_id,
                                                    command_traj, 
                                                    previous_positions,
                                                    previous_rotations,
                                                    next_positions,
                                                    next_rotations,
                                                    images=images,
                                                    tokenizer=self.tokenizer)



                        self.data[split].append(trajectory) 
                        vocab |= trajectory.traj_vocab
        return vocab

    def shuffle_trajectories(self, split=None): 
        """
        shuffle the text annotations across trajectories with the same positions 
        """
        pass 
    


if __name__ == "__main__":
    reader = DatasetReader("blocks_data/devset.json", "blocks_data/devset.json", "blocks_data/devset.json")  
    reader.read_data("train") 
    #reader = DatasetReader("blocks_data/devset.json", "blocks_data/devset.json", "blocks_data/devset.json", batch_by_line=True)
    #reader.read_data("train") 
    
    for trajectory in reader.data["train"]:
        for instance in trajectory: 
            print(instance)   
            sys.exit() 
