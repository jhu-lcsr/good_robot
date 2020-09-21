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

from data import DatasetReader

logger = logging.getLogger(__name__)

def load_data(path):
    all_data = []
    with open(path) as f1:
        for line in f1.readlines():
            all_data.append(json.loads(line))
    return all_data

def get_vocab(data, tokenizer):
    vocab = set()
    for example_line in data: 
        for sent in example_line["notes"]:
            # only use first example for testing 
            sent = sent["notes"][0]
            tokenized = tokenizer(sent)
            tokenized = set(tokenized) 
            vocab |= tokenized
    return vocab 
        

class LanguageTrainer:
    def __init__(self,
                 train_data: List,
                 val_data: List,
                 encoder: LanguageEncoder,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int,
                 device: torch.device,
                 checkpoint_dir: str,
                 num_models_to_keep: int): 
        self.train_data = train_data
        self.val_data   = val_data
        self.encoder = encoder
        self.optimizer = optimizer 
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.num_models_to_keep = num_models_to_keep

        self.loss_fxn = torch.nn.CrossEntropyLoss()
        self.device = device

    def train(self):
        all_accs = []
        max_acc = 0.0 
        for epoch in range(self.num_epochs): 
            acc = self.train_and_validate_one_epoch(epoch)
            # handle checkpointing 
            all_accs.append(acc) 
            is_best = False
            if acc > max_acc:
                is_best = True
                max_acc = acc 
            self.save_model(epoch, is_best) 

    def train_and_validate_one_epoch(self, epoch): 
        print(f"Training epoch {epoch}...") 
        self.encoder.train() 
        for batch_trajectory in tqdm(self.train_data): 
            for batch_instance in batch_trajectory: 
                #self.generate_debugging_image(batch_instance, f"epoch_{epoch}_gold")
                self.optimizer.zero_grad() 
                outputs = self.encoder(batch_instance) 
                loss = self.compute_loss(batch_instance, outputs) 
                loss.backward() 
                self.optimizer.step() 

        print(f"Validating epoch {epoch}...") 
        total_acc = 0.0 
        total = 0 

        self.encoder.eval() 
        for dev_batch_trajectory in tqdm(self.val_data): 
            for dev_batch_instance in dev_batch_trajectory: 
                total_acc += self.validate(dev_batch_instance, epoch) 
                total += 1

        mean_acc = total_acc / total 
        print(f"Epoch {epoch} has acc {mean_acc * 100}") 
        return mean_acc 

    def validate(self, batch_instance, epoch_num): 
        outputs = self.encoder(batch_instance) 
        accuracy = self.compute_localized_accuracy(batch_instance, outputs) 
            
        if epoch_num > 80: 
            self.generate_debugging_image(outputs, f"epoch_{epoch_num}_pred")
            self.generate_debugging_image(batch_instance, f"epoch_{epoch_num}_gold")
            sys.exit() 

        return accuracy

    def compute_localized_accuracy(self, batch_instance, outputs): 
        next_pos = batch_instance["next_position"]
        prev_pos = batch_instance["previous_position_for_acc"]

        gold_pixels_of_interest = next_pos[next_pos != prev_pos]

        values, pred_pixels = torch.max(outputs['next_position'], dim=1) 
        neg_indices = next_pos != prev_pos

        pred_pixels_of_interest = pred_pixels[neg_indices.squeeze(1)]

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

    def compute_accuracy(self, batch_instance, outputs): 
        # compute overlap between predicted and true output 
        gold_pixels = batch_instance["next_position"].to(self.device) 
        values, pred_pixels = torch.max(outputs['next_position'], dim=1) 
        pred_pixels = pred_pixels.reshape(*gold_pixels.shape) 

        # flatten  
        pred_pixels = pred_pixels.reshape(-1) 
        gold_pixels = gold_pixels.reshape(-1) 
        # compare 
        total = gold_pixels.shape[0]
        matching = torch.sum(pred_pixels == gold_pixels).detach().cpu().item() 
        acc = matching/total 
        return acc 

    def generate_debugging_image(self, data, filename):
        # 21 x 4 x 64 x 64 
        next_pos = data["next_position"][0]
        if next_pos.shape[0] == 21:
            # take argmax 
            next_pos_id, next_pos = torch.max(next_pos, dim = 0) 
        next_pos = next_pos.squeeze(0)
        print(f"next pos {next_pos.shape}") 
        # make a logging dir 
        if not os.path.exists(os.path.join(self.checkpoint_dir, "images")):
            os.mkdir(os.path.join(self.checkpoint_dir, "images"))

        xs = np.arange(0, 64, 1)
        zs = np.arange(0, 64, 1)
        # separate plot per depth slice 
        for height in range(4):
            fig = plt.figure(figsize=(12,12))
            ax = fig.gca()
            ax.set_xticks([0, 64])
            ax.set_yticks([0, 64]) 
            ax.set_ylim(0, 64)
            ax.set_xlim(0, 64)
            plt.grid() 

            to_plot_xs, to_plot_zs, to_plot_labels = [], [], []
            for x_pos in xs:
                for z_pos in zs:
                    #label = next_pos[depth, x_pos, y_pos].item() 
                    label = next_pos[x_pos, z_pos, height].item() 
                    if height > 0 and label > 0:
                        print(f"we have a tall block with label {label} at x, z, y: {x_pos, z_pos, height}")
                    # don't plot background 
                    if label != 0: 
                        label = int(label) 
                        to_plot_xs.append(x_pos)
                        to_plot_zs.append(z_pos)
                        to_plot_labels.append(label) 

            ax.plot(to_plot_xs, to_plot_zs, ".")
            for x,z, lab in zip(to_plot_xs, to_plot_zs, to_plot_labels):
                ax.annotate(lab, xy=(x,z), fontsize = 12)

            file_path = os.path.join(self.checkpoint_dir, "images", f"{filename}-{height}.png") 
                
            print(f"saving to {file_path}") 
            plt.savefig(file_path) 

    def compute_loss(self, inputs, outputs):
        pred_image = outputs["next_position"]
        true_image = inputs["next_position"]

        bsz, n_blocks, width, height, depth = pred_image.shape
        true_image = true_image.reshape((bsz, width, height, depth)).long()
        true_image = true_image.to(self.device) 

        loss = self.loss_fxn(pred_image, true_image) 

        return loss 

    def save_model(self, epoch, is_best):
        print(f"Saving checkpoint {epoch}") 
        # get path 
        save_path = os.path.join(self.checkpoint_dir, f"model_{epoch}.th") 
        torch.save(self.encoder.state_dict(), save_path) 
        print(f"Saved checkpoint to {save_path}") 
        # if it's best performance, save extra 
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"best.th") 
            torch.save(self.encoder.state_dict(), best_path) 
            print(f"Updated best model to {best_path} at epoch {epoch}") 

        # remove old models 
        all_paths = glob.glob(os.path.join(self.checkpoint_dir, "model_*th"))
        if len(all_paths) > self.num_models_to_keep:
            to_remove = sorted(all_paths, key = lambda x: int(os.path.basename(x).split(".")[0].split('_')[1]))[0:-self.num_models_to_keep]
            for path in to_remove:
                os.remove(path) 

def main(args):
    # load the data 
    logger.info(f"Reading data from {args.train_path}")
    dataset_reader = DatasetReader(args.train_path,
                                   args.val_path,
                                   None,
                                   batch_by_line = True)

    train_vocab = dataset_reader.read_data("train") 
    dev_vocab = dataset_reader.read_data("dev") 
    
    # construct the vocab and tokenizer 
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)
    
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
    # construct image encoder 
    image_encoder = ImageEncoder(input_dim = 2, 
                                 n_layers = args.conv_num_layers,
                                 factor = args.conv_factor,
                                 dropout = args.dropout)

    # construct image and language fusion module 
    fuser = ConcatFusionModule(image_encoder.output_dim, encoder.hidden_dim) 
    # construct image decoder 
    output_module = DeconvolutionalNetwork(input_channels  = fuser.output_dim, 
                                           num_blocks = args.num_blocks,
                                           num_layers = args.deconv_num_layers,
                                           factor = args.deconv_factor,
                                           dropout = args.dropout) 

    if args.cuda is not None:
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    device = torch.device(device)  
    print(f"Training on device {device}") 
    # put it all together into one module 
    encoder = LanguageEncoder(image_encoder = image_encoder, 
                              embedder = embedder, 
                              encoder = encoder, 
                              fuser = fuser, 
                              output_module = output_module,
                              device = device) 

    #encoder = encoder.to(torch.device(device))
    # construct optimizer 
    optimizer = torch.optim.Adam(encoder.parameters())

    try:
        os.mkdir(args.checkpoint_dir)
    except FileExistsError:
        # file exists
        try:
            assert(len(glob.glob(os.path.join(args.checkpoint_dir, "*.th"))) == 0)
        except AssertionError:
            raise AssertionError(f"Output directory {args.checkpoint_dir} non-empty, will not overwrite!") 

    # construct trainer 
    trainer = LanguageTrainer(train_data = dataset_reader.data["train"], 
                              val_data = dataset_reader.data["dev"], 
                              encoder = encoder,
                              optimizer = optimizer, 
                              num_epochs = args.num_epochs,
                              device = device,
                              checkpoint_dir = args.checkpoint_dir,
                              num_models_to_keep = args.num_models_to_keep) 
    trainer.train() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument("--train-path", type=str, default = "blocks_data/trainset_v2.json", help="path to train data")
    parser.add_argument("--val-path", default = "blocks_data/devset.json", type=str, help = "path to dev data" )
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
    # misc
    parser.add_argument("--output-type", type=str, default="mask")
    parser.add_argument("--dropout", type=float, default=0.2) 
    parser.add_argument("--cuda", type=int, default=None) 
    parser.add_argument("--checkpoint-dir", type=str, default="models/language_pretrain")
    parser.add_argument("--num-models-to-keep", type=int, default = 5) 
    parser.add_argument("--num-epochs", type=int, default=3) 

    args = parser.parse_args()
    main(args) 

