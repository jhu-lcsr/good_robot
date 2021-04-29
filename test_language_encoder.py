import json 
import argparse
from typing import List, Dict
import glob
import os 

import torch
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import logging 
from tqdm import tqdm 
from matplotlib import pyplot as plt
import numpy as np

from image_encoder import ImageEncoder, DeconvolutionalNetwork
from language import LanguageEncoder, ConcatFusionModule
from encoders import LSTMEncoder
from language_embedders import RandomEmbedder
from mlp import MLP 

from data import DatasetReader

logger = logging.getLogger(__name__)


class LanguageTester:
    def __init__(self,
                 test_data: List,
                 encoder: LanguageEncoder,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int,
                 device: torch.device,
                 checkpoint_dir: str,
                 num_models_to_keep: int,
                 generate_after_n: int): 
        self.test_data   = val_data
        self.encoder = encoder
        self.optimizer = optimizer 
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.num_models_to_keep = num_models_to_keep
        self.generate_after_n = generate_after_n

    def evaluate(self):
        total_acc = 0.0 
        total = 0 
        total_block_acc = 0.0 
        self.encoder.eval() 
        for b, dev_batch_trajectory in tqdm(enumerate(self.val_data)): 
            for i, dev_batch_instance in enumerate(dev_batch_trajectory): 
                pixel_acc, block_acc = self.validate(dev_batch_instance, epoch, b, i) 
                total_acc += pixel_acc
                total_block_acc += block_acc
                total += 1

        mean_acc = total_acc / total 
        mean_block_acc = total_block_acc / total
        print(f"Test-time pixel acc {mean_acc * 100}, block acc {mean_block_acc * 100}") 
        return mean_acc 

    def validate(self, batch_instance, do_generate=False, batch_num=None, instance_num=None):
        outputs = self.encoder(batch_instance) 
        accuracy = self.compute_localized_accuracy(batch_instance, outputs) 
        block_accuracy = self.compute_block_accuracy(batch_instance, outputs) 
            
        if do_generate:
            self.generate_debugging_image(outputs, f"{batch_num}_{instance_num}_pred")
            self.generate_debugging_image(batch_instance, f"{batch_num}_{instance_num}_gold")

        return accuracy, block_accuracy

    def compute_localized_accuracy(self, batch_instance, outputs): 
        next_pos = batch_instance["next_position"]
        prev_pos = batch_instance["previous_position_for_acc"]

        gold_pixels_of_interest = next_pos[next_pos != prev_pos]

        values, pred_pixels = torch.max(outputs['next_position'], dim=1) 
        neg_indices = next_pos != prev_pos
        pred_pixels_of_interest = pred_pixels[neg_indices.squeeze(-1)]

        # flatten  
        pred_pixels = pred_pixels_of_interest.reshape(-1).detach().cpu()
        gold_pixels = gold_pixels_of_interest.reshape(-1).detach().cpu()

        # compare 
        total = gold_pixels.shape[0]

        matching = torch.sum(pred_pixels == gold_pixels).item() 
        try:
            acc = matching/total 
        except ZeroDivisionError:
            acc = 0.0 
        return acc 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument("--test-path", default = "blocks_data/devset.json", type=str, help = "path to dev data" )
    parser.add_argument("--num-blocks", type=int, default=20) 
    # language embedder 
    parser.add_argument("--embedder", type=str, default="random", choices = ["random", "glove"])
    parser.add_argument("--embedding-dim", type=int, default=300) 
    # language encoder
    parser.add_argument("--encoder", type=str, default="lstm", choices = ["lstm", "transformer"])
    parser.add_argument("--encoder-hidden-dim", type=int, default=128) 
    parser.add_argument("--encoder-num-layers", type=int, default=2) 
    parser.add_argument("--bidirectional", action="store_true") 
    # image encoder 
    parser.add_argument("--conv-factor", type=int, default = 4) 
    parser.add_argument("--conv-num-layers", type=int, default=2) 
    # image decoder 
    parser.add_argument("--deconv-factor", type=int, default = 2) 
    parser.add_argument("--deconv-num-layers", type=int, default=2) 
    # block mlp
    parser.add_argument("--mlp-hidden-dim", type=int, default = 128) 
    parser.add_argument("--mlp-num-layers", type=int, default = 3) 
    parser.add_argument("--mlp-dropout", type=float, default = 0.20) 
    # misc
    parser.add_argument("--output-type", type=str, default="mask")
    parser.add_argument("--dropout", type=float, default=0.2) 
    parser.add_argument("--cuda", type=int, default=None) 
    parser.add_argument("--checkpoint-dir", type=str, default="models/language_pretrain")
    parser.add_argument("--num-models-to-keep", type=int, default = 5) 
    parser.add_argument("--num-epochs", type=int, default=3) 
    parser.add_argument("--generate-after-n", type=int, default=10) 

    args = parser.parse_args()
    main(args) 

