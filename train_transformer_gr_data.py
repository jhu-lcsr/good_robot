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
from einops import rearrange 
import logging 
from tqdm import tqdm 
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
import numpy as np
import torch.autograd.profiler as profiler
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from allennlp.training.scheduler import Scheduler 
from allennlp.training.learning_rate_schedulers import NoamLR
import pandas as pd 

from transformer import TransformerEncoder, ResidualTransformerEncoder, image_to_tiles, tiles_to_image
from metrics import  MSEMetric, AccuracyMetric, GoodRobotTransformerTeleportationMetric
from language_embedders import RandomEmbedder, GloveEmbedder, BERTEmbedder
from data import DatasetReader, GoodRobotDatasetReader
from train_language_encoder import get_free_gpu, load_data, get_vocab, LanguageTrainer, FlatLanguageTrainer
from train_transformer import TransformerTrainer

logger = logging.getLogger(__name__)

class GoodRobotTransformerTrainer(TransformerTrainer): 
    def __init__(self,
                 train_data: List,
                 val_data: List,
                 encoder: TransformerEncoder,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Scheduler, 
                 num_epochs: int,
                 num_blocks: int, 
                 device: torch.device,
                 checkpoint_dir: str,
                 num_models_to_keep: int,
                 generate_after_n: int,
                 resolution: int = 64, 
                 patch_size: int = 8,
                 block_size: int = 4, 
                 output_type: str = "per-pixel", 
                 depth: int = 7,
                 score_type: str = "acc",
                 best_epoch: int = -1,
                 seed: int = 12, 
                 zero_weight: float = 0.05,
                 next_weight: float = 1.0,
                 prev_weight: float = 1.0,
                 do_regression: bool = False,
                 do_reconstruction: bool = False):
        super(GoodRobotTransformerTrainer, self).__init__(train_data=train_data,
                                                 val_data=val_data,
                                                 encoder=encoder,
                                                 optimizer=optimizer,
                                                 scheduler=scheduler,
                                                 num_epochs=num_epochs,
                                                 num_blocks=num_blocks,
                                                 device=device,
                                                 checkpoint_dir=checkpoint_dir,
                                                 num_models_to_keep=num_models_to_keep,
                                                 generate_after_n=generate_after_n,
                                                 score_type=score_type,
                                                 patch_size=patch_size,
                                                 block_size=block_size,
                                                 output_type=output_type,
                                                 resolution=resolution, 
                                                 depth=depth, 
                                                 best_epoch=best_epoch,
                                                 seed=seed,
                                                 zero_weight=zero_weight,
                                                 next_weight=next_weight,
                                                 prev_weight=prev_weight,
                                                 do_reconstruction=do_reconstruction,
                                                 do_regression=do_regression)

        self.teleportation_metric = GoodRobotTransformerTeleportationMetric(block_size=block_size,
                                                                            image_size = resolution,
                                                                            patch_size = patch_size)


    def train_and_validate_one_epoch(self, epoch): 
        print(f"Training epoch {epoch}...") 
        self.encoder.train() 
        skipped = 0
        for b, batch_instance in tqdm(enumerate(self.train_data)): 
            self.optimizer.zero_grad() 
            outputs = self.encoder(batch_instance) 
            #next_outputs, prev_outputs = self.encoder(batch_instance) 
            # skip bad examples 
            if outputs is None:
                skipped += 1
                continue
            if self.output_type == "per-pixel": 
                loss = self.compute_weighted_loss(batch_instance, outputs, (epoch + 1) * (b+1)) 
            elif self.output_type == "per-patch": 
                loss = self.compute_patch_loss(batch_instance, outputs, self.next_to_prev_weight) 
            elif self.output_type == "patch-softmax":
                loss = self.compute_xent_loss(batch_instance, outputs) 
            else:
                raise AssertionError("must have output in ['per-pixel', 'per-patch', 'patch-softmax']") 
            loss.backward() 
            self.optimizer.step() 
            it = (epoch + 1) * (b+1) 
            self.scheduler.step_batch(it) 

        print(f"skipped {skipped} examples") 
        print(f"Validating epoch {epoch}...") 
        total_prev_acc, total_next_acc = 0.0, 0.0
        total = 0 
        total_block_acc = 0.0 
        total_tele_score = 0.0
        total_mse = 0.0 
        total_prev_recon, total_next_recon = 0.0, 0.0

        self.encoder.eval() 
        for b, dev_batch_instance in tqdm(enumerate(self.val_data)): 
            #prev_pixel_acc, block_acc = self.validate(dev_batch_instance, epoch, b, 0) 
            score_dict = self.validate(dev_batch_instance, epoch, b, 0) 
            total_prev_acc += score_dict['prev_f1']
            total_next_acc += score_dict['next_f1']
            total += 1

        mean_next_acc = total_next_acc / total 
        mean_prev_acc = total_prev_acc / total 
        print(f"Epoch {epoch} has next pixel F1 {mean_next_acc * 100} prev F1 {mean_prev_acc * 100}")
        if self.score_type == "acc":
            return (mean_next_acc + mean_prev_acc)/2, -1.0
        else:
            raise AssertionError(f"invalid score type {self.score_type}")

    def compute_weighted_loss(self, inputs, outputs, it):
        """
        compute per-pixel for all pixels, with additional loss term for only foreground pixels (where true label is 1) 
        """
        pred_next_image = outputs["next_position"]
        true_next_image = inputs["next_pos_for_pred"]
        bsz, n_blocks, width, height, depth = pred_next_image.shape

        pred_next_image = pred_next_image.squeeze(-1)
        true_next_image = true_next_image.squeeze(-1).squeeze(-1)
        true_next_image = true_next_image.long().to(self.device) 

        next_pixel_loss = self.weighted_xent_loss_fxn(pred_next_image, true_next_image) 

        pred_prev_image = outputs["prev_position"]
        true_prev_image = inputs["prev_pos_for_pred"]

        pred_prev_image = pred_prev_image.squeeze(-1)
        true_prev_image = true_prev_image.squeeze(-1).squeeze(-1)
        true_prev_image = true_prev_image.long().to(self.device) 

        prev_pixel_loss = self.weighted_xent_loss_fxn(pred_prev_image, true_prev_image)  

        total_loss = next_pixel_loss + prev_pixel_loss 
        print(f"loss {total_loss.item()}")

        return total_loss

    def compute_patch_loss(self, inputs, outputs, next_to_prev_weight = [1.0, 1.0]):
        """
        compute per-patch for each patch 
        """
        bsz, __, w, h = inputs['prev_pos_input'].shape 

        pred_next_image = outputs["next_position"]
        pred_prev_image = outputs["prev_position"]

        true_next_image = image_to_tiles(inputs["next_pos_for_pred"].reshape(bsz, 1, w, h), self.patch_size) 
        true_prev_image = image_to_tiles(inputs["prev_pos_for_pred"].reshape(bsz, 1, w, h), self.patch_size) 

        # binarize patches
        prev_sum_image = torch.sum(true_prev_image, dim = 2, keepdim=True) 
        prev_patches = torch.zeros_like(prev_sum_image)
        next_sum_image = torch.sum(true_next_image, dim = 2, keepdim=True) 
        next_patches = torch.zeros_like(next_sum_image)
        # any patch that has a 1 pixel in it gets 1 
        prev_patches[prev_sum_image != 0] = 1
        next_patches[next_sum_image != 0] = 1

        pred_prev_image = pred_prev_image.squeeze(-1)
        pred_next_image = pred_next_image.squeeze(-1)
        prev_patches = prev_patches.squeeze(-1).to(self.device).long()
        next_patches = next_patches.squeeze(-1).to(self.device).long()  

        pred_prev_image = rearrange(pred_prev_image, 'b n c -> b c n')
        pred_next_image = rearrange(pred_next_image, 'b n c -> b c n')

        prev_pixel_loss = self.weighted_xent_loss_fxn(pred_prev_image, prev_patches)  
        next_pixel_loss = self.weighted_xent_loss_fxn(pred_next_image, next_patches) 
        next_weight = next_to_prev_weight[0] 
        prev_weight = next_to_prev_weight[1] 

        total_loss = next_weight * next_pixel_loss + prev_weight * prev_pixel_loss 
        print(f"loss {total_loss.item()}")

        if self.do_regression:
            pred_pos = outputs["next_pos_xyz"].reshape(-1) 
            true_pos = inputs["next_pos_for_regression"].reshape(-1).to(self.device)
            reg_loss = self.reg_loss_fxn(pred_pos, true_pos) 
            total_loss += reg_loss 

        if self.do_reconstruction:
            # do state reconstruction from image input for previous and next image 
            true_next_image_recon = image_to_tiles(inputs["next_pos_for_acc"].reshape(bsz, 1, w, h), self.patch_size) 
            true_prev_image_recon = image_to_tiles(inputs["prev_pos_for_acc"].reshape(bsz, 1, w, h), self.patch_size)       
            # take max of each patch so that even mixed patches count as having a block 
            true_next_image_recon, __ = torch.max(true_next_image_recon, dim=2) 
            true_prev_image_recon, __ = torch.max(true_prev_image_recon, dim=2) 

            pred_next_image_recon = outputs["next_per_patch_class"]
            pred_prev_image_recon = outputs["prev_per_patch_class"]
            bsz, n = true_next_image_recon.shape 
            pred_next_image_recon  = pred_next_image_recon.reshape(bsz * n, 21)
            pred_prev_image_recon  = pred_prev_image_recon.reshape(bsz * n, 21)
            
            true_next_image_recon = true_next_image_recon.reshape(-1).to(pred_next_image_recon.device).long()
            true_prev_image_recon = true_prev_image_recon.reshape(-1).to(pred_next_image_recon.device).long() 

            prev_loss = self.xent_loss_fxn(pred_prev_image_recon, true_prev_image_recon)
            next_loss = self.xent_loss_fxn(pred_next_image_recon, true_next_image_recon)
            total_loss += prev_loss + next_loss

        return total_loss

    def compute_xent_loss(self, inputs, outputs): 
        """ 
        instead of bce against each patch, one distribution over all patches 
        """
        bsz, __, w, h = inputs['prev_pos_input'].shape 
        pred_next_image = outputs["next_position"]
        pred_prev_image = outputs["prev_position"]

        pred_next_image = pred_next_image.reshape((bsz, -1))
        pred_prev_image = pred_prev_image.reshape((bsz, -1))

        true_next_image = image_to_tiles(inputs["next_pos_for_pred"].reshape(bsz, 1, w, h), self.patch_size) 
        true_prev_image = image_to_tiles(inputs["prev_pos_for_pred"].reshape(bsz, 1, w, h), self.patch_size) 
        # binarize patches
        prev_sum_image = torch.sum(true_prev_image, dim = 2, keepdim=True) 
        prev_patches = torch.zeros_like(prev_sum_image)
        next_sum_image = torch.sum(true_next_image, dim = 2, keepdim=True) 
        next_patches = torch.zeros_like(next_sum_image)
        # any patch that has a 1 pixel in it gets 1 
        prev_patches[prev_sum_image != 0] = 1
        next_patches[next_sum_image != 0] = 1
        # get single patch index (for now) 
        prev_patches_max = torch.argmax(prev_patches, dim = 1).reshape(-1)
        next_patches_max = torch.argmax(next_patches, dim = 1).reshape(-1) 

        prev_patches_max = prev_patches_max.to(pred_prev_image.device)
        next_patches_max = next_patches_max.to(pred_next_image.device)
        prev_loss = self.xent_loss_fxn(pred_prev_image, prev_patches_max)  
        next_loss = self.xent_loss_fxn(pred_next_image, next_patches_max) 

        total_loss = prev_loss + next_loss 
        print(f"loss {total_loss.item()}")

        return total_loss

    def generate_debugging_image(self, true_img, true_loc, pred_data, out_path, caption):
        caption = self.wrap_caption(caption)
        c = pred_data.shape[0]

        cmap = plt.get_cmap("Reds")
        if c == 2:
            pred_data = pred_data[1,:,:]

        fig, ax = plt.subplots(2,2, figsize=(16,16))

        # gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])
        text_ax = ax[0,1]
        text_ax.axis([0, 1, 0, 1])
        text_ax.text(0.2, 0.02, caption, fontsize = 12)
        text_ax.axis("off") 

        props = dict(boxstyle='round', 
                     facecolor='wheat', alpha=0.5)
        text_ax.text(0.05, 0.95, caption, wrap=True, fontsize=14,
            verticalalignment='top', bbox=props)
        # img_ax = plt.subplot(gs[2])
        img_ax = ax[1,0]
        w = int(40 * (self.resolution / 224))
        true_img = true_img.detach().cpu().numpy().astype(int)[:,:,0:3]
        img_ax.imshow(true_img)
        location = true_loc - int(w/2)
        rect = patches.Rectangle(location, w, w ,linewidth=3,edgecolor='w',facecolor='none')
        img_ax.add_patch(rect)

        # pred_ax = plt.subplot(gs[0], figsize=(6,6))
        pred_ax = ax[0,0]
        xs = np.arange(0, self.resolution, 1)
        zs = np.arange(0, self.resolution, 1)

        ticks = [i for i in range(0, self.resolution + 16, 16)]
        pred_ax.set_xticks(ticks)
        pred_ax.set_yticks(ticks) 
        pred_ax.set_ylim(0, self.resolution)
        pred_ax.set_xlim(0, self.resolution)
        plt.grid() 
        to_plot_xs_prob, to_plot_zs_prob, to_plot_probs = [], [], []
        for x_pos in xs:
            for z_pos in zs:
                prob = pred_data[x_pos, z_pos].item()
                to_plot_zs_prob.append(self.resolution - x_pos)
                to_plot_xs_prob.append(z_pos)
                to_plot_probs.append(prob)

        squares = []
        for x,z, lab in zip(to_plot_xs_prob, to_plot_zs_prob, to_plot_probs):
            rgba = list(cmap(lab))
            # make opaque
            rgba[-1] = 0.4
            sq = matplotlib.patches.Rectangle((x,z), width = 1, height = 1, color = rgba)
            pred_ax.add_patch(sq)

        file_path =  f"{out_path}.png"
        print(f"saving to {file_path}") 
        plt.savefig(file_path) 
        plt.close() 

    def validate(self, batch_instance, epoch_num, batch_num, instance_num): 
        self.encoder.eval() 
        outputs = self.encoder(batch_instance) 
        prev_position = outputs['prev_position']
        next_position = outputs['next_position']

        prev_position = tiles_to_image(prev_position, self.patch_size, output_type="per-patch", upsample=True) 
        next_position = tiles_to_image(next_position, self.patch_size, output_type="per-patch", upsample=True) 
        # f1 metric 
        prev_p, prev_r, prev_f1 = self.compute_f1(batch_instance["prev_pos_for_pred"].squeeze(-1), prev_position) 
        next_p, next_r, next_f1 = self.compute_f1(batch_instance["next_pos_for_pred"].squeeze(-1), next_position) 
        # block accuracy metric 
        tele_metric_data  = self.compute_teleportation_metric(batch_instance["pairs"], prev_position, next_position)

        if epoch_num > self.generate_after_n: 
            for i in range(outputs["next_position"].shape[0]):
                output_path = self.checkpoint_dir.joinpath(f"batch_{batch_num}").joinpath(f"instance_{i}")
                output_path.mkdir(parents = True, exist_ok=True)
                command = batch_instance["command"][i]
                command = [x for x in command if x != "<PAD>"]
                command = " ".join(command) 
                next_pos = batch_instance["next_pos_for_acc"][i]
                prev_pos = batch_instance["prev_pos_for_acc"][i]
                 
                self.generate_debugging_image(next_pos,
                                             batch_instance['pairs'][i].next_location,
                                             next_position[i], 
                                             output_path.joinpath("next"),
                                             caption = command) 

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

        return {"next_f1": next_f1, 
                "prev_f1": prev_f1} 

    def compute_f1(self, true_pos, pred_pos):
        eps = 1e-8
        values, pred_pixels = torch.max(pred_pos, dim=1) 
        gold_pixels = true_pos 
        pred_pixels = pred_pixels.unsqueeze(1) 

        pred_pixels = pred_pixels.detach().cpu().float() 
        gold_pixels = gold_pixels.detach().cpu().float() 

        total_pixels = sum(pred_pixels.shape) 

        true_pos = torch.sum(pred_pixels * gold_pixels).item() 
        true_neg = torch.sum((1-pred_pixels) * (1 - gold_pixels)).item() 
        false_pos = torch.sum(pred_pixels * (1 - gold_pixels)).item() 
        false_neg = torch.sum((1-pred_pixels) * gold_pixels).item() 
        precision = true_pos / (true_pos + false_pos + eps) 
        recall = true_pos / (true_pos + false_neg + eps) 
        f1 = 2 * (precision * recall) / (precision + recall + eps) 
        return precision, recall, f1

    def compute_teleportation_metric(self, pairs, pred_pos, next_pos):
        for pair, ppos, npos in zip(pairs, pred_pos, next_pos): 
            to_ret = self.teleportation_metric.get_metric(pair, ppos, npos)
        return to_ret 



def main(args):
    device = "cpu"
    if args.cuda is not None:
        free_gpu_id = get_free_gpu()
        if free_gpu_id > -1:
            device = f"cuda:{free_gpu_id}"
            #device = "cuda:0"

    device = torch.device(device)  
    print(f"On device {device}") 
    test = torch.ones((1))
    test = test.to(device) 

    # load the data 
    dataset_reader = GoodRobotDatasetReader(path_or_obj=args.path,
                                            split_type=args.split_type,
                                            task_type=args.task_type,
                                            augment_by_flipping = args.augment_by_flipping,
                                            augment_by_rotating = args.augment_by_rotating, 
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

    depth = 1
    encoder_cls = ResidualTransformerEncoder if args.encoder_type == "ResidualTransformerEncoder" else TransformerEncoder
    encoder_kwargs = dict(image_size = args.resolution,
                          patch_size = args.patch_size, 
                          language_embedder = embedder, 
                          n_layers_shared = args.n_shared_layers,
                          n_layers_split  = args.n_split_layers,
                          n_classes = 2,
                          channels = args.channels,
                          n_heads = args.n_heads,
                          hidden_dim = args.hidden_dim,
                          ff_dim = args.ff_dim,
                          dropout = args.dropout,
                          embed_dropout = args.embed_dropout,
                          output_type = args.output_type, 
                          positional_encoding_type = args.pos_encoding_type,
                          device = device,
                          log_weights = args.test,
                          init_scale = args.init_scale, 
                          do_regression = False,
                          do_reconstruction = False) 
    if args.encoder_type == "ResidualTransformerEncoder":
        encoder_kwargs["do_residual"] = args.do_residual 
    # Initialize encoder 
    encoder = encoder_cls(**encoder_kwargs)

    if args.cuda is not None:
        encoder = encoder.cuda(device) 
    print(encoder) 
    # construct optimizer 
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learn_rate) 
    # scheduler
    scheduler = NoamLR(optimizer, model_size = args.hidden_dim, warmup_steps = args.warmup, factor = args.lr_factor) 

    best_epoch = -1
    block_size = int((args.resolution * 4)/64) 
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
            encoder = encoder.to("cpu") 
            state_dict = torch.load(pathlib.Path(args.checkpoint_dir).joinpath("best.th"), map_location='cpu')
            
            encoder.load_state_dict(state_dict, strict=True)  
            encoder = encoder.cuda(device) 
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


        num_blocks = 1
        # construct trainer 
        trainer = GoodRobotTransformerTrainer(train_data = dataset_reader.data["train"], 
                              val_data = dataset_reader.data["dev"], 
                              encoder = encoder,
                              optimizer = optimizer, 
                              scheduler = scheduler, 
                              num_epochs = args.num_epochs,
                              num_blocks = num_blocks,
                              device = device,
                              checkpoint_dir = args.checkpoint_dir,
                              num_models_to_keep = args.num_models_to_keep,
                              generate_after_n = args.generate_after_n, 
                              score_type=args.score_type,
                              depth = depth, 
                              resolution = args.resolution, 
                              output_type = args.output_type, 
                              patch_size = args.patch_size,
                              block_size = block_size, 
                              best_epoch = best_epoch,
                              seed = args.seed,
                              zero_weight = args.zero_weight,
                              next_weight = args.next_weight,
                              prev_weight = args.prev_weight,
                              do_regression = False,
                              do_reconstruction = False) 
        trainer.train() 

    else:
        # test-time, load best model  
        print(f"loading model weights from {args.checkpoint_dir}") 
        #state_dict = torch.load(pathlib.Path(args.checkpoint_dir).joinpath("best.th"))
        #encoder.load_state_dict(state_dict, strict=True)  
        encoder = encoder.to("cpu") 
        state_dict = torch.load(pathlib.Path(args.checkpoint_dir).joinpath("best.th"), map_location='cpu')
        
        encoder.load_state_dict(state_dict, strict=True)  
        encoder = encoder.cuda(device) 

        if "test" in dataset_reader.data.keys():
            eval_data = dataset_reader.data['test']
            out_path = "test_metrics.json"
        else:
            eval_data = dataset_reader.data['dev']
            out_path = "val_metrics.json"

        eval_trainer = GoodRobotTransformerTrainer(train_data = dataset_reader.data["train"], 
                                   val_data = eval_data, 
                                   encoder = encoder,
                                   optimizer = None, 
                                   scheduler = None, 
                                   num_epochs = 0, 
                                   num_blocks = 1, 
                                   device = device,
                                   resolution = args.resolution, 
                                   output_type = args.output_type, 
                                   checkpoint_dir = args.checkpoint_dir,
                                   patch_size = args.patch_size,
                                   block_size = block_size,
                                   num_models_to_keep = 0, 
                                   seed = args.seed,
                                   generate_after_n = args.generate_after_n,
                                   score_type=args.score_type,
                                   do_regression = False, 
                                   do_reconstruction = False) 
        print(f"evaluating") 
        eval_trainer.evaluate(out_path)


if __name__ == "__main__":
    np.random.seed(12)
    torch.manual_seed(12)

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
    parser.add_argument("--task-type", type=str, choices = ["rows", "stacks", "rows-and-stacks"],
                        default="rows-and-stacks") 
    parser.add_argument("--leave-out-color", type=str, default=None) 
    parser.add_argument("--augment-by-flipping", action="store_true")
    parser.add_argument("--augment-by-rotating", action="store_true")
    parser.add_argument("--augment-language", action="store_true")
    parser.add_argument("--overfit", action = "store_true")
    # language embedder 
    parser.add_argument("--embedder", type=str, default="random", choices = ["random", "glove", "bert-base-cased", "bert-base-uncased"])
    parser.add_argument("--embedding-file", type=str, help="path to pretrained glove embeddings")
    parser.add_argument("--embedding-dim", type=int, default=300) 
    # transformer parameters 
    parser.add_argument("--encoder-type", type=str, default="TransformerEncoder", choices = ["TransformerEncoder", "ResidualTransformerEncoder"], help = "choice of dual-stream transformer encoder or one that bases next prediction on previous transformer representation")
    parser.add_argument("--pos-encoding-type", type = str, default="learned") 
    parser.add_argument("--patch-size", type=int, default = 8)  
    parser.add_argument("--n-shared-layers", type=int, default = 6) 
    parser.add_argument("--n-split-layers", type=int, default = 2) 
    parser.add_argument("--n-classes", type=int, default = 2) 
    parser.add_argument("--n-heads", type= int, default = 8) 
    parser.add_argument("--hidden-dim", type= int, default = 512)
    parser.add_argument("--ff-dim", type = int, default = 1024) 
    parser.add_argument("--dropout", type=float, default=0.2) 
    parser.add_argument("--embed-dropout", type=float, default=0.2) 
    parser.add_argument("--output-type", type=str, choices = ["per-pixel", "per-patch", "patch-softmax"], default='per-pixel')
    parser.add_argument("--do-residual", action = "store_true", help = "set to residually connect unshared and next prediction in ResidualTransformerEncoder")
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
    parser.add_argument("--seed", type=int, default=12) 

    args = parser.parse_args() 

    main(args) 
