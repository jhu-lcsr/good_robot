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
from language_embedders import RandomEmbedder, GloveEmbedder
from unet_module import BaseUNet, UNetWithLanguage, UNetWithBlocks
from unet_shared import SharedUNet
from mlp import MLP 
from losses import ScheduledWeightedCrossEntropyLoss

from data import DatasetReader
from train_language_encoder import get_free_gpu, load_data, get_vocab, LanguageTrainer, FlatLanguageTrainer

logger = logging.getLogger(__name__)

class UNetLanguageTrainer(FlatLanguageTrainer): 
    def __init__(self,
                 train_data: List,
                 val_data: List,
                 encoder: SharedUNet,
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
        super(UNetLanguageTrainer, self).__init__(train_data,
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

        weight = torch.tensor([zero_weight, 1.0-zero_weight]).to(device) 
        total_steps = num_epochs * len(train_data) 
        print(f"total steps {total_steps}") 
        #self.weighted_xent_loss_fxn = ScheduledWeightedCrossEntropyLoss(start_weight = 0.50, 
        #                                                                max_weight = 0.01,
        #                                                                num_steps = total_steps/2)

        self.weighted_xent_loss_fxn = torch.nn.CrossEntropyLoss(weight = weight) 
        #self.weighted_xent_loss_fxn = kornia.losses.DiceLoss()
        #self.weighted_xent_loss_fxn = kornia.losses.FocalLoss(0.25, gamma=2.0, reduction='mean') 
        #self.weighted_xent_loss_fxn = BootstrappedCE()
        self.xent_loss_fxn = torch.nn.CrossEntropyLoss()
        self.fore_loss_fxn = torch.nn.CrossEntropyLoss(ignore_index=0)


    def train_and_validate_one_epoch(self, epoch): 
        print(f"Training epoch {epoch}...") 
        self.encoder.train() 
        skipped = 0
        for b, batch_instance in tqdm(enumerate(self.train_data)): 

            self.optimizer.zero_grad() 
            next_outputs, prev_outputs = self.encoder(batch_instance) 
            # skip bad examples 
            if prev_outputs is None:
                skipped += 1
                continue
            loss = self.compute_weighted_loss(batch_instance, next_outputs, prev_outputs, (epoch + 1) * (b+1)) 
            #loss = self.compute_loss(batch_instance, next_outputs, prev_outputs) 
            loss.backward() 
            self.optimizer.step() 

        print(f"skipped {skipped} examples") 
        print(f"Validating epoch {epoch}...") 
        total_prev_acc, total_next_acc = 0.0, 0.0
        total = 0 
        total_block_acc = 0.0 

        self.encoder.eval() 
        for b, dev_batch_instance in tqdm(enumerate(self.val_data)): 
            next_pixel_acc, prev_pixel_acc, block_acc = self.validate(dev_batch_instance, epoch, b, 0) 
            total_prev_acc += prev_pixel_acc
            total_next_acc += next_pixel_acc
            total_block_acc += block_acc
            total += 1

        mean_next_acc = total_next_acc / total 
        mean_prev_acc = total_prev_acc / total 
        mean_block_acc = total_block_acc / total
        print(f"Epoch {epoch} has next pixel acc {mean_next_acc * 100} prev acc {mean_prev_acc * 100}, block acc {mean_block_acc * 100}") 
        return (mean_next_acc + mean_prev_acc)/2, mean_block_acc 

    def compute_loss(self, inputs, next_outputs, prev_outputs):
        """
        compute per-pixel for all pixels, with additional loss term for only foreground pixels (where true label is 1) 
        """
        pred_next_image = next_outputs["next_position"]
        true_next_image = inputs["next_pos_for_pred"]
        pred_prev_image = prev_outputs["next_position"]
        true_prev_image = inputs["prev_pos_for_pred"]

        bsz, n_blocks, width, height, depth = pred_prev_image.shape
        true_next_image = true_next_image.reshape((bsz, width, height, depth)).long()
        true_prev_image = true_prev_image.reshape((bsz, width, height, depth)).long()
        true_next_image = true_next_image.to(self.device) 
        true_prev_image = true_prev_image.to(self.device) 
        
        if self.compute_block_dist:
            pred_next_block_logits = next_outputs["pred_block_logits"] 
            true_next_block_idxs = inputs["block_to_move"]
            true_next_block_idxs = true_next_block_idxs.to(self.device).long().reshape(-1) 
            # TODO (elias): for now just do as auxiliary task 
            next_pixel_loss = self.xent_loss_fxn(pred_next_image, true_next_image) 
            prev_pixel_loss = self.xent_loss_fxn(pred_prev_image, true_prev_image) 
            next_foreground_loss = self.fore_loss_fxn(pred_next_image, true_next_image) 
            prev_foreground_loss = self.fore_loss_fxn(pred_prev_image, true_prev_image) 
            # loss per block
            block_loss = self.xent_loss_fxn(pred_next_block_logits, true_next_block_idxs) 
            total_loss = next_pixel_loss + prev_pixel_loss + next_foreground_loss + prev_foreground_loss + block_loss 
        else:
            prev_pixel_loss = self.xent_loss_fxn(pred_prev_image, true_prev_image) 
            next_pixel_loss = self.xent_loss_fxn(pred_next_image, true_next_image) 
            prev_foreground_loss = self.fore_loss_fxn(pred_prev_image, true_prev_image) 
            next_foreground_loss = self.fore_loss_fxn(pred_next_image, true_next_image) 

            total_loss = next_pixel_loss + prev_pixel_loss + next_foreground_loss +  prev_foreground_loss

        print(f"loss {total_loss.item()}")

        return total_loss


    def compute_weighted_loss(self, inputs, next_outputs, prev_outputs, it):
        """
        compute per-pixel for all pixels, with additional loss term for only foreground pixels (where true label is 1) 
        """
        pred_next_image = next_outputs["next_position"]
        true_next_image = inputs["next_pos_for_pred"]
        pred_prev_image = prev_outputs["next_position"]
        true_prev_image = inputs["prev_pos_for_pred"]

        bsz, n_blocks, width, height, depth = pred_prev_image.shape
        pred_prev_image = pred_prev_image.squeeze(-1)
        pred_next_image = pred_next_image.squeeze(-1)
        true_next_image = true_next_image.squeeze(-1).squeeze(-1)
        true_prev_image = true_prev_image.squeeze(-1).squeeze(-1)
        true_next_image = true_next_image.long().to(self.device) 
        true_prev_image = true_prev_image.long().to(self.device) 

        prev_pixel_loss = self.weighted_xent_loss_fxn(pred_prev_image, true_prev_image)  
        next_pixel_loss = self.weighted_xent_loss_fxn(pred_next_image, true_next_image) 

        total_loss = next_pixel_loss + prev_pixel_loss 
        print(f"loss {total_loss.item()}")

        return total_loss


    def validate(self, batch_instance, epoch_num, batch_num, instance_num): 
        self.encoder.eval() 
        next_outputs, prev_outputs = self.encoder(batch_instance) 

        prev_p, prev_r, prev_f1 = self.compute_f1(batch_instance["prev_pos_for_pred"], prev_outputs["next_position"])
        next_p, next_r, next_f1 = self.compute_f1(batch_instance["next_pos_for_pred"], next_outputs["next_position"]) 
        if self.compute_block_dist:
            block_accuracy = self.compute_block_accuracy(batch_instance, next_outputs) 
        else:
            block_accuracy = -1.0
            
        if epoch_num > self.generate_after_n: 
            for i in range(next_outputs["next_position"].shape[0]):
                output_path = self.checkpoint_dir.joinpath(f"batch_{batch_num}").joinpath(f"instance_{i}")
                output_path.mkdir(parents = True, exist_ok=True)
                command = batch_instance["command"][i]
                command = [x for x in command if x != "<PAD>"]
                command = " ".join(command) 

                next_pos = batch_instance["next_pos_for_acc"][i]
                
                # TODO: remove fake 
                # fake next 
                #fake_next = batch_instance["next_pos_for_pred"][i].reshape(1, 64, 64, 1)
                #fake_next = torch.cat([torch.zeros_like(fake_next), fake_next], dim=0)
                #fake_pred = batch_instance["prev_pos_for_pred"][i].reshape(1, 64, 64, 1) 
                #fake_pred = torch.cat([torch.zeros_like(fake_pred), fake_pred], dim=0)

                self.generate_debugging_image(next_pos,
                                             next_outputs["next_position"][i], 
                                             output_path.joinpath("next"),
                                             caption = command)

                prev_pos = batch_instance["prev_pos_for_acc"][i]
                self.generate_debugging_image(prev_pos, 
                                              prev_outputs["next_position"][i], 
                                              output_path.joinpath("prev"),
                                              caption = command) 

        return next_f1, prev_f1, block_accuracy

    def compute_f1(self, true_pos, pred_pos):
        eps = 1e-8
        values, pred_pixels = torch.max(pred_pos, dim=1) 
        gold_pixels = true_pos 
        pred_pixels = pred_pixels.unsqueeze(-1) 

        pred_pixels = pred_pixels.detach().cpu()
        gold_pixels = gold_pixels.detach().cpu()

        total_pixels = sum(pred_pixels.shape) 

        true_pos = torch.sum(pred_pixels * gold_pixels).item() 
        true_neg = torch.sum((1-pred_pixels) * (1 - gold_pixels)).item() 
        false_pos = torch.sum(pred_pixels * (1 - gold_pixels)).item() 
        false_neg = torch.sum((1-pred_pixels) * gold_pixels).item() 
        precision = true_pos / (true_pos + false_pos + eps) 
        recall = true_pos / (true_pos + false_neg + eps) 
        f1 = 2 * (precision * recall) / (precision + recall + eps) 
        return precision, recall, f1

    def compute_localized_accuracy(self, true_pos, pred_pos, waste): 
        values, pred_pixels = torch.max(pred_pos, dim=1) 
        pred_pixels = pred_pixels.unsqueeze(-1) 

        gold_pixels_ones = true_pos[true_pos == 1]
        pred_pixels_ones = pred_pixels[true_pos == 1]

        # flatten  
        pred_pixels_ones = pred_pixels_ones.reshape(-1).detach().cpu()
        gold_pixels_ones = gold_pixels_ones.reshape(-1).detach().cpu()

        # compare 
        total_foreground = gold_pixels_ones.shape[0]
        matching_foreground = torch.sum(pred_pixels_ones == gold_pixels_ones).item() 
        try:
            foreground_acc = matching_foreground/total_foreground
        except ZeroDivisionError:
            foreground_acc = 0.0 

        gold_pixels_zeros = true_pos[true_pos == 0]
        pred_pixels_zeros = pred_pixels[true_pos == 0]
        # flatten  
        pred_pixels_zeros = pred_pixels_zeros.reshape(-1).detach().cpu()
        gold_pixels_zeros = gold_pixels_zeros.reshape(-1).detach().cpu()

        total_background = gold_pixels_zeros.shape[0]
        matching_background = torch.sum(pred_pixels_zeros == gold_pixels_zeros).item() 
        try:
            background_acc = matching_background/total_background
        except ZeroDivisionError:
            background_acc = 0.0 

        #print(f"foreground {foreground_acc} background {background_acc}") 
        return (foreground_acc + background_acc ) / 2

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
    elif args.embedder == "glove":
        embedder = GloveEmbedder(tokenizer, train_vocab, args.embedding_file, args.embedding_dim, trainable=True) 
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


    if args.top_only:
        depth = 1
    else:
        # TODO (elias): confirm this number 
        depth = 7

    unet_kwargs = dict(in_channels = 21,
                     out_channels = args.unet_out_channels, 
                     lang_embedder = embedder,
                     lang_encoder = encoder, 
                     hc_large = args.unet_hc_large,
                     hc_small = args.unet_hc_small,
                     kernel_size = args.unet_kernel_size,
                     stride = args.unet_stride,
                     num_layers = args.unet_num_layers,
                     num_blocks = args.num_blocks,
                     dropout = args.dropout,
                     depth = depth,
                     device=device)


    if args.compute_block_dist:
        unet_kwargs["mlp_num_layers"] = args.mlp_num_layers

    encoder = SharedUNet(**unet_kwargs) 

    if args.cuda is not None:
        encoder= encoder.cuda(device) 
                         
    print(encoder) 
    # construct optimizer 
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learn_rate) 

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
        trainer = UNetLanguageTrainer(train_data = dataset_reader.data["train"], 
                              val_data = dataset_reader.data["dev"], 
                              encoder = encoder,
                              optimizer = optimizer, 
                              num_epochs = args.num_epochs,
                              num_blocks = args.num_blocks,
                              device = device,
                              checkpoint_dir = args.checkpoint_dir,
                              num_models_to_keep = args.num_models_to_keep,
                              generate_after_n = args.generate_after_n, 
                              depth = depth, 
                              best_epoch = best_epoch,
                              zero_weight = args.zero_weight) 
        trainer.train() 

    else:
        # test-time, load best model  
        print(f"loading model weights from {args.checkpoint_dir}") 
        state_dict = torch.load(pathlib.Path(args.checkpoint_dir).joinpath("best.th"))
        encoder.load_state_dict(state_dict, strict=True)  

        eval_trainer = UNetLanguageTrainer(train_data = dataset_reader.data["train"], 
                                   val_data = dataset_reader.data["dev"], 
                                   encoder = encoder,
                                   optimizer = None, 
                                   num_epochs = 0, 
                                   num_blocks = args.num_blocks,
                                   device = device,
                                   checkpoint_dir = args.checkpoint_dir,
                                   num_models_to_keep = 0, 
                                   generate_after_n = 0) 
        print(f"evaluating") 
        eval_trainer.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training 
    parser.add_argument("--test", action="store_true", help="load model and test")
    parser.add_argument("--resume", action="store_true", help="resume training a model")
    # data 
    parser.add_argument("--train-path", type=str, default = "blocks_data/trainset_v2.json", help="path to train data")
    parser.add_argument("--val-path", default = "blocks_data/devset.json", type=str, help = "path to dev data" )
    parser.add_argument("--num-blocks", type=int, default=20) 
    parser.add_argument("--binarize-blocks", action="store_true", help="flag to treat block prediction as binary task instead of num-blocks-way classification") 
    parser.add_argument("--traj-type", type=str, default="flat", choices = ["flat", "trajectory"]) 
    parser.add_argument("--batch-size", type=int, default = 32) 
    parser.add_argument("--max-seq-length", type=int, default = 65) 
    parser.add_argument("--do-filter", action="store_true", help="set if we want to restrict prediction to the block moved") 
    parser.add_argument("--top-only", action="store_true", help="set if we want to train/predict only the top-most slice of the top-down view") 
    # language embedder 
    parser.add_argument("--embedder", type=str, default="random", choices = ["random", "glove"])
    parser.add_argument("--embedding-file", type=str, help="path to pretrained glove embeddings")
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
    # unet parameters 
    parser.add_argument("--share-level", type=str, help="share the weights between predicting previous and next position") 
    parser.add_argument("--unet-out-channels", type=int, default=128)
    parser.add_argument("--unet-hc-large", type=int, default=32)
    parser.add_argument("--unet-hc-small", type=int, default=16) 
    parser.add_argument("--unet-num-layers", type=int, default=5) 
    parser.add_argument("--unet-stride", type=int, default=2) 
    parser.add_argument("--unet-kernel-size", type=int, default=5) 
    # misc
    parser.add_argument("--dropout", type=float, default=0.2) 
    parser.add_argument("--cuda", type=int, default=None) 
    parser.add_argument("--learn-rate", type=float, default = 0.001) 
    parser.add_argument("--checkpoint-dir", type=str, default="models/language_pretrain")
    parser.add_argument("--num-models-to-keep", type=int, default = 5) 
    parser.add_argument("--num-epochs", type=int, default=3) 
    parser.add_argument("--generate-after-n", type=int, default=10) 
    parser.add_argument("--zero-weight", type=float, default = 0.05, help = "weight for loss weighting negative vs positive examples") 

    args = parser.parse_args()
    main(args) 

