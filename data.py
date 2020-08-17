import os 
import sys
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from spacy.tokenizer import Tokenizer 
from spacy.lang.en import English


# what is the image size: depth x 64 x 64 
# block distribution by id: depth x 64 x 64 x n_blocks 
# operation distribution by id: depth x 64 x 64 x n_ops 

class BaseTrajectory:
    def __init__(self,
                 line_id: int, 
                 commands: List,
                 initial_position: Tuple[float],
                 initial_rotation: Tuple[float],
                 positions: List[Tuple[float]], 
                 rotations: List[Tuple[float]], 
                 images: List[str]):
        self.line_id = line_id
        self.commands = commands
        self.initial_position = initial_position
        self.initial_rotation = initial_rotation
        self.positions = positions
        self.rotations = rotations
        self.images = images

class SimpleTrajectory(BaseTrajectory):
    def __init__(self,
                 line_id: int,
                 commands: List[str], 
                 initial_position: Tuple[float],
                 initial_rotation: Tuple[float],
                 positions: List[Tuple[float]], 
                 rotations: List[Tuple[float]], 
                 images: List[str],
                 tokenizer: Tokenizer):
        super(SimpleTrajectory, self).__init__(line_id,
                                         commands, 
                                         initial_position,
                                         initial_rotation,
                                         positions, 
                                         rotations, 
                                         images) 
        self.tokenizer = tokenizer
        # commands is a list of #timestep text strings 
        self.commands = self.tokenize(commands)

    def tokenize(self, commands): 
        for timestep, command in enumerate(commands):
            commands[timestep] = list(self.tokenizer(command))
        return commands 

class BatchedTrajectory(BaseTrajectory): 
    def __init__(self,
                 line_id: int,
                 commands: List[List[str]], 
                 initial_position: Tuple[float],
                 initial_rotation: Tuple[float],
                 positions: List[Tuple[float]], 
                 rotations: List[Tuple[float]], 
                 images: List[str],
                 tokenizer: Tokenizer):
        super(BatchedTrajectory, self).__init__(line_id,
                                                commands,
                                                initial_position,
                                                initial_rotation,
                                                positions,
                                                rotations,
                                                images) 
        # commands is now a 9xlen matrix 
        self.tokenizer = tokenizer
        self.commands = self.pad_commands(self.tokenize(commands)) 

    def tokenize(self, commands): 
        for annotator_idx, command_list in enumerate(commands):
            for timestep, command in enumerate(command_list):
                commands[annotator_idx][timestep] = list(self.tokenizer(command))
        return commands 

    def pad_commands(self, commands): 
        """
        pad commands in a batch to have the same length
        """
        max_len = 0
        # get max length 
        for annotator_idx, command_list in enumerate(commands):
            for command_seq in command_list:
                if len(command_seq) > max_len:
                    max_len = len(command_seq) 

        # pad to max length 
        for annotator_idx, command_list in enumerate(commands):
            for i, command_seq in enumerate(command_list): 
                commands[annotator_idx][i] += ["<PAD>" for i in range(max_len - len(command_seq))]

        return commands 

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

    def read_data(self, split):
        """
        extract trajectories for training and evaluation 


        """
        if split not in self.paths:
            raise AssertionError(f"split {split} not valid. Options are {self.path.keys()}") 

        with open(self.paths[split]) as f1:
            for line_id, line in enumerate(f1.readlines()): 
                line_data = json.loads(line) 

                positions = line_data["states"]
                rotations = line_data["rotations"]
                commands = line_data["notes"]
                # TODO: add images  
                # split off initial and subsequent positions and rotations 
                initial_position, positions = positions[0], positions[1:]
                initial_rotation, rotations = rotations[0], rotations[1:]
                command_trajectories = [[] for i in range(len(commands[0]['notes']))]
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
                                                   initial_position,
                                                   initial_rotation,
                                                   positions,
                                                   rotations,
                                                   images = None,
                                                   tokenizer = self.tokenizer) 
                    self.data[split].append(trajectory)  

                else:
                    for command_traj in command_trajectories:
                        trajectory = SimpleTrajectory(line_id,
                                                    command_traj, 
                                                    initial_position,
                                                    initial_rotation,
                                                    positions,
                                                    rotations,
                                                    images=None,
                                                    tokenizer=self.tokenizer)



                        self.data[split].append(trajectory) 

    def shuffle_trajectories(self, split=None): 
        """
        shuffle the text annotations across trajectories with the same positions 
        """
        pass 
    


if __name__ == "__main__":
    reader = DatasetReader("blocks_data/devset.json", "blocks_data/devset.json", "blocks_data/devset.json")  
    reader.read_data("train") 
    print(len(reader.data["train"]) )
    reader = DatasetReader("blocks_data/devset.json", "blocks_data/devset.json", "blocks_data/devset.json", batch_by_line=True)
    reader.read_data("train") 
    print(len(reader.data["train"]) )

