
import json 
import argparse
from typing import List, Dict
import glob
import os 
import pathlib
import pdb 
import subprocess 
from io import StringIO

import torch
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import logging 
from tqdm import tqdm 
from matplotlib import pyplot as plt
import numpy as np
import torch.autograd.profiler as profiler
from torch.nn import functional as F
import pandas as pd 
import kornia 


from encoders import LSTMEncoder
from language_only import LanguageEncoder
from language_embedders import RandomEmbedder
from unet_module import BaseUNet, UNetWithLanguage, UNetWithBlocks
from unet_shared import SharedUNet
from mlp import MLP 

from data import DatasetReader
from train_language_encoder import get_free_gpu, load_data, get_vocab, LanguageTrainer, FlatLanguageTrainer


class LanguageAloneTrainer(FlatLanguageTrainer): 
    def __init__(self,
                 train_data: List,
                 val_data: List,
                 encoder: LanguageEncoder,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int,
                 num_blocks: int, 
                 device: torch.device,
                 checkpoint_dir: str,
                 num_models_to_keep: int,
                 generate_after_n: int,
                 depth: int = 7,
                 best_epoch: int = -1,
                 zero_weight: float = 0.05):
        super(LanguageAloneTrainer, self).__init__(train_data,
                                                  val_data,
                                                  encoder,
                                                  optimizer,
                                                  num_epochs,
                                                  num_blocks,
                                                  device,
                                                  checkpoint_dir,
                                                  num_models_to_keep,
                                                  generate_after_n,
                                                  depth, 
                                                  best_epoch)

        self.xent_loss_fxn = torch.nn.CrossEntropyLoss()

    def train_and_validate_one_epoch(self, epoch): 
        print(f"Training epoch {epoch}...") 
        self.encoder.train() 
        skipped = 0
        for b, batch_instance in tqdm(enumerate(self.train_data)): 

            self.optimizer.zero_grad() 
            lang_outputs = self.encoder(batch_instance) 
            loss = self.compute_loss(batch_instance, lang_outputs) 
            loss.backward() 
            self.optimizer.step() 

        print(f"skipped {skipped} examples") 
        print(f"Validating epoch {epoch}...") 
        total = 0 
        total_block_acc = 0.0 

        self.encoder.eval() 
        for b, dev_batch_instance in tqdm(enumerate(self.val_data)): 
            block_acc = self.validate(dev_batch_instance, epoch, b, 0) 
            total_block_acc += block_acc
            total += 1

        mean_block_acc = total_block_acc / total
        print(f"Epoch {epoch} has block acc {mean_block_acc * 100}") 
        return mean_block_acc, 0.0

    def compute_loss(self, inputs, lang_outputs):
        pred_next_block_logits = lang_outputs["pred_block_logits"] 
        true_next_block_idxs = inputs["block_to_move"]
        true_next_block_idxs = true_next_block_idxs.to(self.device).long().reshape(-1) 
        block_loss = self.xent_loss_fxn(pred_next_block_logits, true_next_block_idxs) 
        return block_loss 

    def validate(self, batch_instance, epoch_num, batch_num, instance_num): 
        self.encoder.eval() 
        #pdb.set_trace() 
        lang_outputs= self.encoder(batch_instance) 
        block_accuracy = self.compute_block_accuracy(batch_instance, lang_outputs) 
        return block_accuracy

def main(args):
    if args.binarize_blocks:
        args.num_blocks = 1

    device = "cpu"
    if args.cuda is not None:
        free_gpu_id = get_free_gpu()
        if free_gpu_id > -1:
            device = f"cuda:{free_gpu_id}"

    device = torch.device(device)  
    print(f"On device {device}") 
    test = torch.ones((1))
    test = test.to(device) 

    # load the data 
    dataset_reader = DatasetReader(args.train_path,
                                   args.val_path,
                                   None,
                                   batch_by_line = args.traj_type != "flat",
                                   traj_type = args.traj_type,
                                   batch_size = args.batch_size,
                                   max_seq_length = args.max_seq_length,
                                   do_filter = args.do_filter,
                                   top_only = args.top_only,
                                   binarize_blocks = args.binarize_blocks)  

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if not args.test:
        print(f"Reading data from {args.train_path}")
        train_vocab = dataset_reader.read_data("train") 
        try:
            os.mkdir(checkpoint_dir)
        except FileExistsError:
            pass
        with open(checkpoint_dir.joinpath("vocab.json"), "w") as f1:
            json.dump(list(train_vocab), f1) 
    else:
        print(f"Reading vocab from {checkpoint_dir}") 
        with open(checkpoint_dir.joinpath("vocab.json")) as f1:
            train_vocab = json.load(f1) 
        
    print(f"Reading data from {args.val_path}")
    dev_vocab = dataset_reader.read_data("dev") 

    print(f"got data")  
    # construct the vocab and tokenizer 
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)
    print(f"constructing model...")  
    # get the embedder from args 
    if args.embedder == "random":
        embedder = RandomEmbedder(tokenizer, train_vocab, args.embedding_dim, trainable=True)
    else:
        raise NotImplementedError(f"No embedder {args.embedder}") 
    # get the encoder from args  
    if args.encoder == "lstm":
        encoder = LSTMEncoder(input_dim = args.embedding_dim,
                              hidden_dim = args.encoder_hidden_dim,
                              num_layers = args.encoder_num_layers,
                              dropout = args.dropout,
                              bidirectional = args.bidirectional) 
    else:
        raise NotImplementedError(f"No encoder {args.encoder}") # construct the model 

    encoder = LanguageEncoder(embedder,
                              encoder,
                              device ) 

    if args.cuda is not None:
        encoder= encoder.cuda(device) 
                         
    print(encoder) 
    # construct optimizer 
    optimizer = torch.optim.Adam(encoder.parameters()) 
    #optimizer = torch.optim.SGD(encoder.parameters(), lr = 0.01 ) 

    best_epoch = -1
    if not args.test:
        if not args.resume:
            try:
                os.mkdir(args.checkpoint_dir)
            except FileExistsError:
                # file exists
                try:
                    assert(len(glob.glob(os.path.join(args.checkpoint_dir, "*.th"))) == 0)
                except AssertionError:
                    raise AssertionError(f"Output directory {args.checkpoint_dir} non-empty, will not overwrite!") 
        else:
            # resume from pre-trained 
            state_dict = torch.load(pathlib.Path(args.checkpoint_dir).joinpath("best.th"))
            encoder.load_state_dict(state_dict, strict=True)  
            # get training info 
            best_checkpoint_data = json.load(open(pathlib.Path(args.checkpoint_dir).joinpath("best_training_state.json")))
            print(f"best_checkpoint_data {best_checkpoint_data}") 
            best_epoch = best_checkpoint_data["epoch"]

        # save arg config to checkpoint_dir
        with open(pathlib.Path(args.checkpoint_dir).joinpath("config.json"), "w") as f1:
            json.dump(args.__dict__, f1) 


        # construct trainer 
        trainer = LanguageAloneTrainer(train_data = dataset_reader.data["train"], 
                              val_data = dataset_reader.data["dev"], 
                              encoder = encoder,
                              optimizer = optimizer, 
                              num_epochs = args.num_epochs,
                              num_blocks = args.num_blocks,
                              device = device,
                              checkpoint_dir = args.checkpoint_dir,
                              num_models_to_keep = args.num_models_to_keep,
                              generate_after_n = 110, 
                              depth = 0, 
                              best_epoch = best_epoch,
                              zero_weight = 0) 
        trainer.train() 

    else:
        # test-time, load best model  
        print(f"loading model weights from {args.checkpoint_dir}") 
        state_dict = torch.load(pathlib.Path(args.checkpoint_dir).joinpath("best.th"))
        encoder.load_state_dict(state_dict, strict=True)  

        trainer = LanguageAloneTrainer(train_data = dataset_reader.data["train"], 
                              val_data = dataset_reader.data["dev"], 
                              encoder = encoder,
                              optimizer = optimizer, 
                              num_epochs = args.num_epochs,
                              num_blocks = args.num_blocks,
                              device = device,
                              checkpoint_dir = args.checkpoint_dir,
                              num_models_to_keep = args.num_models_to_keep,
                              generate_after_n = 110, 
                              depth = 0, 
                              best_epoch = best_epoch,
                              zero_weight = 0) 
        print(f"evaluating") 
        eval_trainer.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training 
    parser.add_argument("--test", action="store_true", help="load model and test")
    parser.add_argument("--resume", action="store_true", help="resume training a model")
    # data 
    parser.add_argument("--train-path", type=str,  help="path to train data")
    parser.add_argument("--val-path",  type=str, help = "path to dev data" )
    parser.add_argument("--num-blocks", type=int, default=20) 
    parser.add_argument("--binarize-blocks", action="store_true", help="flag to treat block prediction as binary task instead of num-blocks-way classification") 
    parser.add_argument("--traj-type", type=str, default="flat", choices = ["flat", "trajectory"]) 
    parser.add_argument("--batch-size", type=int, default = 32) 
    parser.add_argument("--max-seq-length", type=int, default = 65) 
    parser.add_argument("--do-filter", action="store_true", help="set if we want to restrict prediction to the block moved") 
    parser.add_argument("--top-only", action="store_true", help="set if we want to train/predict only the top-most slice of the top-down view") 
    # language embedder 
    parser.add_argument("--embedder", type=str, default="random", choices = ["random", "glove"])
    parser.add_argument("--embedding-dim", type=int, default=300) 
    # language encoder
    parser.add_argument("--encoder", type=str, default="lstm", choices = ["lstm", "transformer"])
    parser.add_argument("--encoder-hidden-dim", type=int, default=128) 
    parser.add_argument("--encoder-num-layers", type=int, default=2) 
    parser.add_argument("--bidirectional", action="store_true") 
    # block mlp
    parser.add_argument("--compute-block-dist", action="store_true") 
    parser.add_argument("--mlp-hidden-dim", type=int, default = 128) 
    parser.add_argument("--mlp-num-layers", type=int, default = 3) 
    # misc
    parser.add_argument("--dropout", type=float, default=0.2) 
    parser.add_argument("--cuda", type=int, default=None) 
    parser.add_argument("--checkpoint-dir", type=str, default="models/language_pretrain")
    parser.add_argument("--num-models-to-keep", type=int, default = 5) 
    parser.add_argument("--num-epochs", type=int, default=3) 

    args = parser.parse_args()
    main(args) 

