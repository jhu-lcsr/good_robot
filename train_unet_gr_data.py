import json 
from jsonargparse import ArgumentParser, ActionConfigFile 
import yaml 
from typing import List, Dict
import glob
import os 
import pathlib
import pdb 
import subprocess 
import copy
from io import StringIO
from collections import defaultdict

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

from encoders import LSTMEncoder
from language_embedders import RandomEmbedder, GloveEmbedder, BERTEmbedder
from unet_module import BaseUNet, UNetWithLanguage, UNetWithBlocks
from unet_shared import SharedUNet
from metrics import GoodRobotUNetTeleportationMetric
from mlp import MLP 
from losses import ScheduledWeightedCrossEntropyLoss

from data import DatasetReader, GoodRobotDatasetReader
from train_language_encoder import get_free_gpu, load_data, get_vocab, LanguageTrainer, FlatLanguageTrainer
from train_unet import UNetLanguageTrainer

logger = logging.getLogger(__name__)

class GoodRobotUNetLanguageTrainer(UNetLanguageTrainer): 
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
                 resolution: int = 64, 
                 depth: int = 7,
                 best_epoch: int = -1,
                 zero_weight: float = 0.05,
                 seed: int = 12,
                 do_reconstruction: bool = False):
        super(GoodRobotUNetLanguageTrainer, self).__init__(train_data = train_data,
                                                          val_data = val_data,
                                                          encoder = encoder,
                                                          optimizer = optimizer,
                                                          num_epochs = num_epochs,
                                                          num_blocks = num_blocks,
                                                          device = device,
                                                          checkpoint_dir = checkpoint_dir,
                                                          num_models_to_keep = num_models_to_keep,
                                                          generate_after_n = generate_after_n,
                                                          resolution = resolution, 
                                                          depth = depth, 
                                                          best_epoch = best_epoch,
                                                          zero_weight=zero_weight)

        self.teleportation_metric = GoodRobotUNetTeleportationMetric(block_size = 4, image_size = self.resolution) 
        self.do_reconstruction = do_reconstruction
        self.set_all_seeds(seed) 

    def set_all_seeds(self, seed):
        np.random.seed(seed) 
        torch.manual_seed(seed) 
        torch.backends.cudnn.deterministic = True

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
        total_tele_score = 0.0
        total_recon_score = 0.0 

        self.encoder.eval() 
        for b, dev_batch_instance in tqdm(enumerate(self.val_data)): 
            score_dict = self.validate(dev_batch_instance, epoch, b, 0) 
            total_prev_acc += score_dict['prev_f1']
            total_next_acc += score_dict['next_f1']
            total_block_acc += score_dict['block_acc']
            total_tele_score += score_dict['tele_dist']
            total_recon_score += score_dict['prev_recon_acc']

            total += 1

        mean_next_acc = total_next_acc / total 
        mean_prev_acc = total_prev_acc / total 
        mean_block_acc = total_block_acc / total
        mean_tele_score = total_tele_score / total 
        mean_recon_score = total_recon_score / total 
        print(f"Epoch {epoch} has next pixel F1 {mean_next_acc * 100} prev F1 {mean_prev_acc * 100}, block acc {mean_block_acc * 100} teleportation score: {mean_tele_score}, recon_score {mean_recon_score}") 
        return (mean_next_acc + mean_prev_acc)/2, mean_block_acc 

    def compute_recon_loss(self, inputs, prev_outputs):
        """
        compute per-pixel for all pixels
        """
        pred_prev_image = prev_outputs["prev_position"]
        true_prev_image = inputs["prev_pos_for_pred"]

        bsz, n_blocks, width, height, depth = pred_prev_image.shape
        true_next_image = true_next_image.reshape((bsz, width, height, depth)).long()
        true_prev_image = true_prev_image.reshape((bsz, width, height, depth)).long()
        true_next_image = true_next_image.to(self.device) 
        true_prev_image = true_prev_image.to(self.device) 
        prev_pixel_loss = self.xent_loss_fxn(pred_prev_image, true_prev_image) 

        return prev_pixel_loss 


    def compute_weighted_loss(self, inputs, next_outputs, prev_outputs, it):
        """
        compute per-pixel for all pixels, with additional loss term for only foreground pixels (where true label is 1) 
        """
        pred_next_image = next_outputs["next_position"]
        true_next_image = inputs["next_pos_for_pred"]
        pred_prev_image = prev_outputs["next_position"]
        true_prev_image = inputs["prev_pos_for_pred"]

        bsz, n_blocks, width, height, depth = pred_prev_image.shape
        pred_prev_image = pred_prev_image.reshape(-1, n_blocks)
        pred_next_image = pred_next_image.reshape(-1, n_blocks)
        true_next_image = true_next_image.reshape(-1)
        true_prev_image = true_prev_image.reshape(-1)
        true_next_image = true_next_image.long().to(self.device) 
        true_prev_image = true_prev_image.long().to(self.device) 

        prev_pixel_loss = self.weighted_xent_loss_fxn(pred_prev_image, true_prev_image)  
        next_pixel_loss = self.weighted_xent_loss_fxn(pred_next_image, true_next_image) 

        if self.do_reconstruction:
            recon_loss = self.compute_recon_loss(inputs, prev_outputs)
        else:
            recon_loss = 0.0
        total_loss = next_pixel_loss + prev_pixel_loss + recon_loss
        print(f"loss {total_loss.item()}")

        return total_loss
        
    def compute_teleportation_metric(self, pairs,  pred_pos, next_pos):
        res = self.teleportation_metric.get_metric(pairs, pred_pos, next_pos)
        return res

    def validate(self, batch_instance, epoch_num, batch_num, instance_num): 
        self.encoder.eval() 
        outputs = self.encoder(batch_instance) 
        next_outputs, prev_outputs = self.encoder(batch_instance) 
        next_position = next_outputs['next_position']
        prev_position = prev_outputs['next_position']

        # f1 metric 
        prev_p, prev_r, prev_f1 = self.compute_f1(batch_instance["prev_pos_for_pred"].squeeze(-1), prev_position) 
        next_p, next_r, next_f1 = self.compute_f1(batch_instance["next_pos_for_pred"].squeeze(-1), next_position) 
        # block accuracy metric 
        # looks like there's some shuffling going on here 
        tele_metric_data = {"distance": [], "block_acc": [], "pred_center": [], "true_center": []}
        
        for i in range(next_position.shape[0]):
            single_tele_dict = self.compute_teleportation_metric(batch_instance["pairs"][i], prev_position[i].detach().clone(), next_position[i].detach().clone())
            tele_metric_data['distance'].append(single_tele_dict['distance'])
            tele_metric_data['block_acc'].append(single_tele_dict['block_acc'])
            tele_metric_data['pred_center'].append(single_tele_dict['pred_center'])
            tele_metric_data['true_center'].append(single_tele_dict['true_center'])

        block_acc = np.mean(tele_metric_data['block_acc'])
        tele_dist = np.mean(tele_metric_data['distance'])

        if epoch_num > self.generate_after_n: 
            for i in range(outputs["next_position"].shape[0]):
                output_path = self.checkpoint_dir.joinpath(f"batch_{batch_num}").joinpath(f"instance_{i}")
                output_path.mkdir(parents = True, exist_ok=True)
                command = batch_instance["command"][i]
                command = [x for x in command if x != "<PAD>"]
                command = " ".join(command) 
                next_pos = batch_instance["next_pos_for_vis"][i]
                prev_pos = batch_instance["prev_pos_for_vis"][i]
                self.generate_debugging_image(next_pos,
                                             batch_instance['pairs'][i].next_location,
                                             next_position[i], 
                                             output_path.joinpath("next"),
                                             caption = command,
                                             pred_center=tele_metric_data["pred_center"][i],
                                             true_center = batch_instance['pairs'][i].next_location) 

                self.generate_debugging_image(prev_pos, 
                                              batch_instance['pairs'][i].prev_location,
                                              prev_position[i], 
                                              output_path.joinpath("prev"),
                                              caption = command) 
                try:
                    with open(output_path.joinpath("attn_weights"), "w") as f1:
                        # for now, just take the last layer 
                        to_dump = {"command": batch_instance['command'][i],
                                   "prev_weight": outputs['prev_attn_weights'][-1][i],
                                   "next_weight": outputs['next_attn_weights'][-1][i]}
                        json.dump(to_dump, f1) 

                except IndexError:
                    # train-time, pass 
                    pass

        prev_recon_acc = 0.0
        if self.do_reconstruction:
            bsz, w, h = batch_instance["prev_pos_for_acc"].shape
            true_prev_image_recon = image_to_tiles(batch_instance["prev_pos_for_acc"].reshape(bsz, 1, w, h), self.patch_size)       
            # take max of each patch so that even mixed patches count as having a block 
            true_prev_image_recon, __ = torch.max(true_prev_image_recon, dim=2) 
            prev_recon_acc = self.reconstruction_metric(true_prev_image_recon,
                                                           outputs['prev_per_patch_class']) 

        return {"next_f1": next_f1, 
                "prev_f1": prev_f1,
                "block_acc": block_acc,
                "tele_dist": tele_dist,
                "prev_recon_acc": prev_recon_acc} 


    def compute_f1(self, true_pos, pred_pos):
        eps = 1e-8
        values, pred_pixels = torch.max(pred_pos, dim=1) 
        gold_pixels = true_pos 
        pred_pixels = pred_pixels.unsqueeze(-1) 

        pred_pixels = pred_pixels.detach().cpu().float() 
        gold_pixels = gold_pixels.detach().cpu().float() 

        bsz, w, h, __, __ = pred_pixels.shape
        pred_pixels = pred_pixels.reshape(bsz, w, h)
        gold_pixels = gold_pixels.reshape(bsz, w, h)
        total_pixels = sum(pred_pixels.shape) 

        true_pos = torch.sum(pred_pixels * gold_pixels).item() 
        true_neg = torch.sum((1-pred_pixels) * (1 - gold_pixels)).item() 
        false_pos = torch.sum(pred_pixels * (1 - gold_pixels)).item() 
        false_neg = torch.sum((1-pred_pixels) * gold_pixels).item() 
        precision = true_pos / (true_pos + false_pos + eps) 
        recall = true_pos / (true_pos + false_neg + eps) 
        f1 = 2 * (precision * recall) / (precision + recall + eps) 
        return precision, recall, f1

def main(args):
    device = "cpu"
    if args.cuda is not None:
        free_gpu_id = get_free_gpu()
        if free_gpu_id > -1:
            device = f"cuda:{free_gpu_id}"

    device = torch.device(device)  
    print(f"On device {device}") 
    test = torch.ones((1))
    test = test.to(device) 

    color_pair = args.color_pair.split(",") if args.color_pair is not None else None 
    dataset_reader = GoodRobotDatasetReader(path_or_obj=args.path,
                                            split_type=args.split_type,
                                            color_pair=color_pair,
                                            task_type=args.task_type,
                                            augment_by_flipping = args.augment_by_flipping,
                                            augment_by_rotating = args.augment_by_rotating, 
                                            augment_with_noise = args.augment_with_noise, 
                                            augment_language = args.augment_language,
                                            leave_out_color = args.leave_out_color,
                                            batch_size=args.batch_size,
                                            max_seq_length=args.max_seq_length,
                                            resolution = args.resolution,
                                            is_bert = "bert" in args.embedder,
                                            overfit=args.overfit) 

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)

    if not args.test:
        train_vocab = dataset_reader.vocab
        with open(checkpoint_dir.joinpath("vocab.json"), "w") as f1:
            json.dump(list(train_vocab), f1) 
    else:
        print(f"Reading vocab from {checkpoint_dir}") 
        with open(checkpoint_dir.joinpath("vocab.json")) as f1:
            train_vocab = json.load(f1) 
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
    elif args.embedder.startswith("bert"): 
        embedder = BERTEmbedder(model_name = args.embedder,  max_seq_len = args.max_seq_length) 
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

    depth = 1
    num_blocks = 1
    unet_kwargs = dict(in_channels = 6,
                     out_channels = args.unet_out_channels, 
                     lang_embedder = embedder,
                     lang_encoder = encoder, 
                     hc_large = args.unet_hc_large,
                     hc_small = args.unet_hc_small,
                     kernel_size = args.unet_kernel_size,
                     stride = args.unet_stride,
                     num_layers = args.unet_num_layers,
                     num_blocks = num_blocks,
                     unet_type = args.unet_type, 
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
        with open(pathlib.Path(args.checkpoint_dir).joinpath("config.yaml"), "w") as f1:
            dump_args = copy.deepcopy(args) 
            # drop stuff we can't serialize 
            del(dump_args.__dict__["cfg"]) 
            del(dump_args.__dict__["__cwd__"]) 
            del(dump_args.__dict__["__path__"]) 
            to_dump = dump_args.__dict__
            # dump 
            yaml.safe_dump(to_dump, f1, encoding='utf-8', allow_unicode=True) 

        # construct trainer 
        trainer = GoodRobotUNetLanguageTrainer(train_data = dataset_reader.data["train"], 
                        val_data = dataset_reader.data["dev"], 
                        encoder = encoder,
                        optimizer = optimizer, 
                        num_epochs = args.num_epochs,
                        num_blocks = num_blocks,
                        device = device,
                        checkpoint_dir = args.checkpoint_dir,
                        num_models_to_keep = args.num_models_to_keep,
                        generate_after_n = args.generate_after_n, 
                        depth = depth, 
                        resolution = args.resolution, 
                        best_epoch = best_epoch,
                        seed = args.seed,
                        zero_weight = args.zero_weight,
                        do_reconstruction = args.do_reconstruction) 
        trainer.train() 

    else:
        # test-time, load best model  
        print(f"loading model weights from {args.checkpoint_dir}") 
        state_dict = torch.load(pathlib.Path(args.checkpoint_dir).joinpath("best.th"))
        encoder.load_state_dict(state_dict, strict=True)  

        if "test" in dataset_reader.data.keys():
            eval_data = dataset_reader.data['test']
            pdb.set_trace() 
            out_path = "test_metrics.json"
        else:
            eval_data = dataset_reader.data['dev']
            out_path = "val_metrics.json"

        eval_trainer = GoodRobotUNetLanguageTrainer(train_data = dataset_reader.data["train"], 
                        val_data = dataset_reader.data["dev"], 
                        encoder = encoder,
                        optimizer = optimizer, 
                        num_epochs = args.num_epochs,
                        num_blocks = num_blocks,
                        device = device,
                        checkpoint_dir = args.checkpoint_dir,
                        num_models_to_keep = args.num_models_to_keep,
                        generate_after_n = args.generate_after_n, 
                        depth = depth, 
                        resolution = args.resolution, 
                        best_epoch = best_epoch,
                        seed = args.seed,
                        zero_weight = args.zero_weight,
                        do_reconstruction = args.do_reconstruction) 
        print(f"evaluating") 
        eval_trainer.evaluate(out_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    # config file 
    parser.add_argument("--cfg", action = ActionConfigFile) 

    # training 
    parser.add_argument("--test", action="store_true", help="load model and test")
    parser.add_argument("--resume", action="store_true", help="resume training a model")
    # data 
    parser.add_argument("--path", type=str, default = "blocks_data/trainset_v2.json", help="path to train data")
    parser.add_argument("--batch-size", type=int, default = 32) 
    parser.add_argument("--max-seq-length", type=int, default = 65) 
    parser.add_argument("--resolution", type=int, help="resolution to discretize input state", default=64) 
    parser.add_argument("--next-weight", type=float, default=1)
    parser.add_argument("--prev-weight", type=float, default=1) 
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--split-type", type=str, choices= ["random", "leave-out-color",
                                                             "train-stack-test-row",
                                                             "train-row-test-stack"],
                                                             default="random")
    parser.add_argument("--color-pair", type=str, default = None, help = "pair of colors to hold out, e.g. red,blue or green,yellow, etc.")
    parser.add_argument("--task-type", type=str, choices = ["rows", "stacks", "rows-and-stacks"],
                        default="rows-and-stacks") 
    parser.add_argument("--leave-out-color", type=str, default=None) 
    parser.add_argument("--augment-by-flipping", action="store_true")
    parser.add_argument("--augment-by-rotating", action="store_true")
    parser.add_argument("--augment-with-noise", action="store_true")
    parser.add_argument("--augment-language", action="store_true")
    parser.add_argument("--overfit", action = "store_true")
    # language embedder 
    parser.add_argument("--embedder", type=str, default="random", choices = ["random", "glove", "bert-base-cased", "bert-base-uncased"])
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
    parser.add_argument("--unet-type", type=str, default="unet_with_attention", help = "type of unet to use") 
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
    parser.add_argument("--do-reconstruction", type=bool, default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=12) 
    args = parser.parse_args()
    main(args) 

