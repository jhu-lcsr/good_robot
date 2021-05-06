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
from metrics import  MSEMetric, AccuracyMetric, F1Metric
from language_embedders import RandomEmbedder, GloveEmbedder, BERTEmbedder
from navigation_data import NavigationDatasetReader, NavigationImageTrajectory, configure_parser
from train_language_encoder import get_free_gpu, load_data, get_vocab, LanguageTrainer, FlatLanguageTrainer
from navigation_transformer import NavigationTransformerEncoder
from train_transformer import TransformerTrainer

logger = logging.getLogger(__name__)

class NavigationTransformerTrainer(TransformerTrainer): 
    def __init__(self,
                 dataset_reader: NavigationDatasetReader,
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
                 batch_size: int = 16, 
                 output_type: str = "per-pixel", 
                 checkpoint_every: int = 64,
                 validation_limit: int = 16, 
                 depth: int = 7,
                 score_type: str = "acc",
                 best_epoch: int = -1,
                 seed: int = 12, 
                 zero_weight: float = 0.05):
        super(NavigationTransformerTrainer, self).__init__(train_data=[],
                                                 val_data=[],
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
                                                 zero_weight=zero_weight) 
        self.f1_metric = F1Metric() 
        self.dataset_reader = dataset_reader
        self.batch_size = batch_size 
        self.checkpoint_every = checkpoint_every
        self.validation_limit = validation_limit

    def split_large_batch(self, batch):
        large_bsz = batch['path_state'].shape[0]
        small_batches = []
        for i in range(0, large_bsz, self.batch_size):
            small_batch = {} 
            for k in batch.keys():
                small_batch[k] = batch[k][i:i+self.batch_size]
            small_batches.append(small_batch)
        return small_batches

    def validate_one_epoch(self, epoch, step, validation_limit):
        print(f"Validating epoch {epoch} step {step}...") 
        total_prev_acc, total_next_acc = 0.0, 0.0
        total = 0 
        self.encoder.eval() 
        for b, dev_batch_instance in enumerate(self.dataset_reader.read("dev", validation_limit)): 
            actual_batches = self.split_large_batch(dev_batch_instance)
            for small_batch in actual_batches:
                score_dict = self.validate(small_batch, epoch, b, 0) 
                total_next_acc += score_dict['next_f1']
                total += 1

        mean_next_acc = total_next_acc / total 
        return mean_next_acc

    def evaluate(self):
        total_acc = 0.0 
        total = 0 
        total_block_acc = 0.0 
        self.encoder.eval() 
        for b, dev_batch_instance in tqdm(enumerate(self.dataset_reader.read("dev", self.validation_limit))): 
            actual_batches = self.split_large_batch(dev_batch_instance)
            for small_batch in actual_batches:
                score_dict = self.validate(small_batch, 10, b, 0) 
                total_acc += score_dict['next_f1']
                total += 1

        mean_acc = total_acc / total 
        print(f"Test-time pixel acc {mean_acc * 100}") 
        return mean_acc 

    def train_and_validate_one_epoch(self, epoch): 
        print(f"Training epoch {epoch}...") 
        self.encoder.train() 
        skipped = 0
        step = 0
        for b, batch_instance in enumerate(self.dataset_reader.read("train")): 
            actual_batches = self.split_large_batch(batch_instance)
            for sb, small_batch in enumerate(actual_batches):
                is_best = False
                self.optimizer.zero_grad() 
                outputs = self.encoder(small_batch) 
                # skip bad examples 
                if outputs is None:
                    skipped += 1
                    continue

                loss = self.compute_patch_loss(small_batch, outputs, self.next_to_prev_weight) 
                loss.backward() 
                self.optimizer.step() 
                it = (epoch + 1) * (step+1) 
                self.scheduler.step_batch(it) 
                #print(f"step: {step+1} checkpoint_every: {self.checkpoint_every}  {(step +1) % self.checkpoint_every}")
                if (step+1) % self.checkpoint_every == 0:
                    step_acc = self.validate_one_epoch(epoch, step, self.validation_limit)
                    print(f"Epoch {epoch} step {step} has next pixel F1 {step_acc * 100:.2f}")
                    if step_acc > self.best_score:
                        is_best = True
                        self.best_score = step_acc

                    self.save_model(f"{epoch}_{step}", is_best) 

                step += 1
        print(f"skipped {skipped} examples") 
        epoch_acc = self.validate_one_epoch(epoch, step, 10 * self.validation_limit) 
        print(f"Epoch {epoch} has next pixel F1 {epoch_acc * 100:.2f}") 
        if self.score_type == "acc":
            return (epoch_acc)/2, -1.0
        else:
            raise AssertionError(f"invalid score type {self.score_type}")

    def compute_patch_loss(self, inputs, outputs, next_to_prev_weight = [1.0, 1.0]):
        """
        compute per-patch for each patch 
        """
        bsz, w, h, __ = inputs['input_image'].shape 

        pred_next_image = outputs["next_position"]

        path_state = inputs['path_state'].reshape(bsz, 1, w, h).float() 
        true_next_image = image_to_tiles(path_state, self.patch_size) 

        # binarize patches
        next_sum_image = torch.sum(true_next_image, dim = 2, keepdim=True) 
        next_patches = torch.zeros_like(next_sum_image)
        # any patch that has a 1 pixel in it gets 1 
        next_patches[next_sum_image != 0] = 1

        pred_next_image = pred_next_image.squeeze(-1)
        next_patches = next_patches.squeeze(-1).to(self.device).long()  

        pred_next_image = rearrange(pred_next_image, 'b n c -> b c n')

        next_pixel_loss = self.weighted_xent_loss_fxn(pred_next_image, next_patches) 

        total_loss = next_pixel_loss 
        print(f"loss {total_loss.item()}")

        return total_loss

    def generate_debugging_image(self, 
                                 true_img, 
                                 path_state, 
                                 pred_path, 
                                 out_path, 
                                 caption = None,
                                 pred_center = None,
                                 true_center = None):
        caption = self.wrap_caption(caption)

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
        #w = int(40 * (self.resolution / 224))
        true_img = true_img.detach().cpu().numpy().astype(float)[:,:,0:3]
        img_ax.imshow(true_img)

        true_path = path_state.detach().numpy()
        true_path = np.tile(true_path.reshape(512, 512, 1), (1,1,3)).astype(float)

        true_ax = ax[0,0]
        true_ax.imshow(true_path)

        pred_path = torch.softmax(pred_path, dim=0)
        pred_path = pred_path[1,:,:]

        pred_path = pred_path.cpu().detach().numpy().reshape(512, 512, 1)
        pred_path = np.tile(pred_path, (1,1,3)).astype(float)

        pred_ax = ax[1,1]
        pred_ax.imshow(pred_path)

        file_path =  f"{out_path}.png"
        print(f"saving to {file_path}") 
        plt.savefig(file_path) 
        plt.close() 

    def validate(self, batch_instance, epoch_num, batch_num, instance_num): 
        self.encoder.eval() 
        outputs = self.encoder(batch_instance) 
        next_position = outputs['next_position']

        next_position = tiles_to_image(next_position, self.patch_size, output_type="per-patch", upsample=True) 
        # f1 metric 
        next_p, next_r, next_f1 = self.f1_metric.compute_f1(batch_instance["path_state"].unsqueeze(-1), next_position) 

        if epoch_num > self.generate_after_n: 
            for i in range(outputs["next_position"].shape[0]):
                output_path = self.checkpoint_dir.joinpath(f"batch_{batch_num}").joinpath(f"instance_{i}")
                output_path.mkdir(parents = True, exist_ok=True)
                command = batch_instance["command"][i]
                command = [x for x in command if x != "<PAD>"]
                command = " ".join(command) 
                image = batch_instance['input_image'][i]
                path_state = batch_instance["path_state"][i]
                pred_path = next_position[i]
                self.generate_debugging_image(image,
                                             path_state,
                                             pred_path,
                                             output_path,
                                             caption = command)

        return {"next_f1": next_f1} 

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

def main(args):
    device = "cpu"
    if args.cuda is not None:
        free_gpu_id = get_free_gpu()
        if free_gpu_id > -1:
            device = f"cuda:{free_gpu_id}"
            #device = "cuda:0"

    device = torch.device(device)  
    print(f"On device {device}") 
    #test = torch.ones((1))
    #test = test.to(device) 

    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)

    dataset_reader = NavigationDatasetReader(dir = args.data_dir,
                                             out_path = args.out_path,
                                             path_width = args.path_width,
                                             read_limit = args.read_limit, 
                                             batch_size = args.batch_size, 
                                             max_len = args.max_len,
                                             tokenizer = tokenizer,
                                             shuffle = args.shuffle,
                                             overfit = args.overfit, 
                                             is_bert = "bert" in args.embedder) 

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir() 

    if not args.test:
        with open(dataset_reader.path_dict['train'].joinpath("vocab.json")) as f1:
            train_vocab = json.load(f1)
        with open(checkpoint_dir.joinpath("vocab.json"), "w") as f1:
            json.dump(list(train_vocab), f1) 
    else:
        print(f"Reading vocab from {checkpoint_dir}") 
        with open(checkpoint_dir.joinpath("vocab.json")) as f1:
            train_vocab = json.load(f1) 

    print(f"got data")  

    # construct the vocab and tokenizer 
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
    encoder_cls = NavigationTransformerEncoder
    
    encoder_kwargs = dict(image_size = args.resolution,
                          patch_size = args.patch_size, 
                          language_embedder = embedder, 
                          n_layers = args.n_layers,
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
                          init_scale = args.init_scale) 

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

    else:
        # test-time, load best model  
        print(f"loading model weights from {args.checkpoint_dir}") 
        #state_dict = torch.load(pathlib.Path(args.checkpoint_dir).joinpath("best.th"))
        #encoder.load_state_dict(state_dict, strict=True)  
        encoder = encoder.to("cpu") 
        state_dict = torch.load(pathlib.Path(args.checkpoint_dir).joinpath("best.th"), map_location='cpu')
        
        encoder.load_state_dict(state_dict, strict=True)  
        encoder = encoder.cuda(device) 

    num_blocks = 1
    # construct trainer 
    trainer = NavigationTransformerTrainer(dataset_reader = dataset_reader,
                            encoder = encoder,
                            optimizer = optimizer, 
                            scheduler = scheduler, 
                            num_epochs = args.num_epochs,
                            num_blocks = num_blocks,
                            device = device,
                            checkpoint_dir = args.checkpoint_dir,
                            checkpoint_every = args.checkpoint_every, 
                            validation_limit = args.validation_limit, 
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
                            zero_weight = args.zero_weight) 

    if not args.test:
        trainer.train() 
    else:
        print(f"evaluating") 
        acc = trainer.evaluate()
        print(f"accuracy: {acc}")


if __name__ == "__main__":
    np.random.seed(12)
    torch.manual_seed(12)

    parser = configure_parser()
    args = parser.parse_args() 

    main(args) 

