import numpy as np 
from jsonargparse import ArgumentParser, ActionConfigFile 
import json 
from tqdm import tqdm 
import pdb 
import pathlib 
from matplotlib import pyplot as plt 
import pickle as pkl 
from spacy.tokenizer import Tokenizer 
from spacy.lang.en import English
import torch
from torch.nn import functional as F

nlp = English()

np.random.seed(12) 
torch.manual_seed(12) 

PAD = "<PAD>"

class NavigationImageTrajectory:
    def __init__(self,
                 image_path: np.array,
                 path: np.array,
                 command: str,
                 tokenizer: Tokenizer,
                 max_len: int = 40,
                 image_size: int = 512,
                 width: int = 8):
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.traj_vocab = set()
        self.lengths = []
        self.command = self.tokenize(command)[0:max_len]
        self.path_state = np.zeros((image_size, image_size)).astype(int)
        # convert path to int 
        self.path = path * 100 
        self.path = self.path.astype(int) 
        self.width = width
        self.start_pos = self.path[0]

        for x, y in self.path:
            self.path_state[x-self.width:x+self.width, y-self.width:y+self.width] = 1

        self.tensorize() 

    def tokenize(self, command): 
        # lowercase everything 
        command = [str(x).lower() for x in self.tokenizer(command)]
        self.lengths = [len(command)]
        # add to vocab 
        self.traj_vocab |= set(command) 
        return command

    def tensorize(self):
        #self.image = plt.imread(self.image_path)
        #self.image = torch.tensor(self.image, dtype = torch.long).unsqueeze(0)

        self.path_state = torch.tensor(self.path_state, dtype = torch.uint8).unsqueeze(0)
        self.start_pos = torch.tensor(self.start_pos, dtype = torch.long).unsqueeze(0)

class NavigationDatasetReader: 
    def __init__(self,
                 dir: str,
                 out_path: str,
                 path_width: int = 8,
                 read_limit: int = -1, 
                 batch_size: int = 64,
                 max_len: int = 40,
                 tokenizer: Tokenizer = Tokenizer(nlp.vocab),
                 shuffle: bool = True,
                 is_bert: bool = False,
                 overfit: bool = False):

        self.path_width = path_width 
        self.dir = pathlib.Path(dir)
        self.pkl_dir = self.dir.joinpath("data/simulator_basic/")
        self.image_dir = self.dir.joinpath("configs/env_img/simulator/")
        self.train_json = self.dir.joinpath("configs/train_annotations_6000.json")
        self.test_json = self.dir.joinpath("configs/test_annotations_6000.json")
        self.dev_json = self.dir.joinpath("configs/dev_annotations_6000.json")
        self.trajectory_class = NavigationImageTrajectory
        self.batch_size = batch_size 
        self.shuffle = shuffle 
        self.max_len = max_len 
        self.tokenizer = tokenizer
        self.read_limit = read_limit 
        self.is_bert = is_bert 
        self.overfit = overfit
        self.out_path = pathlib.Path(out_path)
        self.train_out_path = self.out_path.joinpath("train")
        self.dev_out_path = self.out_path.joinpath("dev")
        self.test_out_path = self.out_path.joinpath("test")

        for p in [self.train_out_path, self.dev_out_path, self.test_out_path]:
            if not p.exists():
                p.mkdir() 

        self.path_dict = {"train": self.train_out_path,
                          "test": self.test_out_path,
                          "dev": self.dev_out_path}


    def make_vocab(self):
        with open(self.train_json) as f1:
            data = json.load(f1)
        print(f"reading vocab...")
        for line in tqdm(data):
            try:
                id = line['id']
                pkl_data = pkl.load(open(self.pkl_dir.joinpath(f"supervised_train_data_env_{id}"), "rb"))
                for step in pkl_data:
                    command = step['instruction']
                    command = [str(x).lower() for x in self.tokenizer(command)]
                    self.vocab |= set(command)
            except FileNotFoundError:
                pass

    def preprocess_batches(self):

        vocab = set()
        for name, path in [("train", self.train_json), ("test", self.test_json), ("dev", self.dev_json)]:
            print(f"loading data from {path}")
            with open(path) as f1:
                data = json.load(f1)
            skipped = 0
            if self.read_limit > -1: 
                data = data[0:self.read_limit]

            line_num = 0 
            curr_batch = []
            batch_num = 0
            for line in tqdm(data):
                try:
                    id = line['id']
                    image_data = plt.imread(self.image_dir.joinpath(f"{id}.png"))
                    pkl_data = pkl.load(open(self.pkl_dir.joinpath(f"supervised_train_data_env_{id}"), "rb"))
                    # get unique steps 
                    unique_steps = []
                    all_commands = [step['instruction'] for step in pkl_data]
                    unique_commands = set(all_commands) 
                    if len(unique_commands) > 1: 
                        unique_indices = [all_commands.index(c) for c in unique_commands]
                    else:
                        unique_indices = [0]

                    for step_idx in unique_indices:
                        step = pkl_data[step_idx]
                        assert(int(step['env_id']) == int(id)) 
                        path = step['seg_path']
                        command = step['instruction']
                        image_path = self.image_dir.joinpath(f"{id}.png")
                        if not image_path.exists():
                            continue
                        traj = NavigationImageTrajectory(image_path = image_path,
                                                        path = path,
                                                        command = command,
                                                        width = self.path_width,
                                                        tokenizer = self.tokenizer,
                                                        max_len = self.max_len)
                        if name == "train":
                            vocab |= traj.traj_vocab

                        curr_batch.append(traj)
                        if line_num % self.batch_size == 0:
                            ready_batch = self.batchify(curr_batch)
                            with open(self.path_dict[name].joinpath(f"{batch_num}.pkl"), "wb") as f1:
                                pkl.dump(ready_batch, f1)
                            # TODO: remove after debugging 
                            with open(self.path_dict['train'].joinpath("vocab.json"), "w") as f1:
                                json.dump(list(vocab), f1)
                            batch_num += 1 
                            curr_batch = []

                        line_num += 1
                except FileNotFoundError:
                    skipped += 1
                    continue

            # add last incomplete batch 
            if len(curr_batch)>0:
                ready_batch = self.batchify(curr_batch)
                with open(self.path_dict[name].joinpath(f"{batch_num+1}.pkl"), "wb") as f1:
                    pkl.dump(ready_batch, f1)

            print(f"skipped {skipped} of {len(data)}: {100*skipped/len(data):.2f}%")
        with open(self.path_dict['train'].joinpath("vocab.json"), "w") as f1:
            json.dump(list(vocab), f1)
        #if self.overfit:
        #    self.all_data['train'] = self.all_data['train'][0:self.read_limit]
        #    self.all_data['dev'] = self.all_data['train']

    def batchify(self, batch_as_list): 
        """
        pad and tensorize 
        """
        commands = []
        input_image = []
        path_state = []
        start_position = []
        # get max len 
        if not self.is_bert:
            max_length = min(self.max_len, max([traj.lengths[0] for traj in batch_as_list])) 
        else:
            max_length = self.max_len

        length = []
        image_paths = []
        for idx in range(len(batch_as_list)):
            traj = batch_as_list[idx]

            # trim! 
            if len(traj.command) > max_length:
                traj.command = traj.command[0:max_length]

            length.append(len(traj.command))
            image_paths.append(traj.image_path)
            commands.append(traj.command + [PAD for i in range(max_length - len(traj.command))])
            #input_image.append(traj.image)
            path_state.append(traj.path_state)
            start_position.append(traj.start_pos)

        #input_image = torch.cat(input_image, 0)
        path_state = torch.cat(path_state, 0)
        start_position = torch.cat(start_position, 0)

        return {"command": commands,
                "image_paths": image_paths,
                "path_state": path_state,
                "start_position": start_position,
                "length": length} 

    def pad_command(self, commands, max_len):
        for i, c in enumerate(commands):
            c = c[0:max_len]
            l = len(c)
            c = c + [PAD for i in range(max_len - l)]
            commands[i] = c
        return commands 

    def read(self, split, limit=None):
        path = self.path_dict[split]
        all_batches = path.glob("*.pkl")
        if self.shuffle and split == "train": 
            np.random.shuffle(all_batches) 
        if limit is not None:
            all_batches = list(all_batches)[0:limit] 

        for batch in all_batches: 
            with open(batch, "rb") as f1:
                batch_data = pkl.load(f1)
            image_paths = batch_data['image_paths']
            image_data = [torch.tensor(plt.imread(p), dtype=torch.float64).unsqueeze(0) for p in image_paths]
            batch_data['input_image'] = torch.cat(image_data, dim=0) 
            if self.is_bert:
                batch_data['command'] = self.pad_command(batch_data['command'], self.max_len)

            yield batch_data 

def configure_parser():
    parser = ArgumentParser()
    
    # config file 
    parser.add_argument("--cfg", action = ActionConfigFile) 
    
    # training 
    parser.add_argument("--test", action="store_true", help="load model and test")
    parser.add_argument("--resume", action="store_true", help="resume training a model")
    parser.add_argument("--overfit", action="store_true", help="overfit to training data for development")
    # data 
    parser.add_argument("--data-dir", type=str, default = "/srv/local2/estengel/nav_data/drif_workspace_corl2019", help="path to train data")
    parser.add_argument("--out-path", type=str, default = "/srv/local2/estengel/nav_data/preprocessed", help = "path to write preprocessed batches")
    parser.add_argument("--batch-size", type=int, default = 32) 
    parser.add_argument("--small-batch-size", type=int, default = 8) 
    parser.add_argument("--max-len", type=int, default = 65) 
    parser.add_argument("--resolution", type=int, help="resolution to discretize input state", default=64) 
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--split-type", type=str, choices= ["random", "leave-out-color",
                                                             "train-stack-test-row",
                                                             "train-row-test-stack"],
                                                             default="random")
    parser.add_argument("--shuffle", action = "store_true")
    parser.add_argument("--read-limit", type=int, default=-1)
    parser.add_argument("--path-width", type=int, default=8)
    parser.add_argument("--output-type", type=str, default="per-patch")
    parser.add_argument("--validation-limit", type=int, default=16, help = "how many dev batches to evaluate every n steps ")
    # language embedder 
    parser.add_argument("--embedder", type=str, default="random", choices = ["random", "glove", "bert-base-cased", "bert-base-uncased"])
    parser.add_argument("--embedding-file", type=str, help="path to pretrained glove embeddings")
    parser.add_argument("--embedding-dim", type=int, default=300) 
    # transformer parameters 
    parser.add_argument("--encoder-type", type=str, default="TransformerEncoder", choices = ["TransformerEncoder", "ResidualTransformerEncoder"], help = "choice of dual-stream transformer encoder or one that bases next prediction on previous transformer representation")
    parser.add_argument("--pos-encoding-type", type = str, default="fixed-separate") 
    parser.add_argument("--patch-size", type=int, default = 8)  
    parser.add_argument("--n-layers", type=int, default = 6) 
    parser.add_argument("--n-classes", type=int, default = 2) 
    parser.add_argument("--n-heads", type= int, default = 8) 
    parser.add_argument("--hidden-dim", type= int, default = 512)
    parser.add_argument("--ff-dim", type = int, default = 1024) 
    parser.add_argument("--dropout", type=float, default=0.2) 
    parser.add_argument("--embed-dropout", type=float, default=0.2) 
    parser.add_argument("--pretrained-weights", type=str, default=None, help = "path to best.th file for a pre-trained initialization")
    parser.add_argument("--locality-mask", type=bool, default = False, action='store_true', help="mask image transformer to only attend to nearby regions")
    parser.add_argument("--locality-neighborhood", type=int, default = 5, help="size of the region to attend to in locality masking, extends in each direction from the center point") 
    # misc
    parser.add_argument("--cuda", type=int, default=None) 
    parser.add_argument("--learn-rate", type=float, default = 3e-5) 
    parser.add_argument("--warmup", type=int, default=4000, help = "warmup setps for learn-rate scheduling")
    parser.add_argument("--lr-factor", type=float, default = 1.0, help = "factor for learn-rate scheduling") 
    parser.add_argument("--gamma", type=float, default = 0.7) 
    parser.add_argument("--checkpoint-dir", type=str, default="models/language_pretrain")
    parser.add_argument("--num-models-to-keep", type=int, default = 5) 
    parser.add_argument("--num-epochs", type=int, default=3) 
    parser.add_argument("--generate-after-n", type=int, default=10) 
    parser.add_argument("--score-type", type=str, default="acc", choices = ["acc", "block_acc", "tele_score"])
    parser.add_argument("--zero-weight", type=float, default = 0.05, help = "weight for loss weighting negative vs positive examples") 
    parser.add_argument("--init-scale", type=int, default = 4, help = "initalization scale for transformer weights")
    parser.add_argument("--checkpoint-every", type=int, default=64, help = "save a checkpoint every n training steps")
    parser.add_argument("--seed", type=int, default=12) 
    parser.add_argument("--debug-image-top-k", type=int, default=-1, help = "for generating debugging images, only show the top k regions")
    parser.add_argument("--debug-image-threshold", type=float, default=-1, help = "for generating debugging images, only predicted patches above a fixed threshold")

    return parser

if __name__ == "__main__":
    np.random.seed(12)
    torch.manual_seed(12)

    parser = configure_parser() 
    
    args = parser.parse_args() 

    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)

    dataset_reader = NavigationDatasetReader(dir = args.data_dir,
                                             out_path=args.out_path,
                                             path_width = args.path_width,
                                             read_limit = args.read_limit, 
                                             batch_size = args.batch_size, 
                                             max_len = args.max_len,
                                             tokenizer = tokenizer,
                                             shuffle = args.shuffle,
                                             overfit = args.overfit, 
                                             is_bert = "bert" in args.embedder) 


    dataset_reader.preprocess_batches()







