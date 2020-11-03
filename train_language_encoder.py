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
import pandas as pd 


from image_encoder import ImageEncoder, DeconvolutionalNetwork, DecoupledDeconvolutionalNetwork
from language import LanguageEncoder, ConcatFusionModule, TiledFusionModule
from encoders import LSTMEncoder
from language_embedders import RandomEmbedder
from mlp import MLP 

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
                 num_blocks: int,
                 device: torch.device,
                 checkpoint_dir: str,
                 num_models_to_keep: int,
                 generate_after_n: int,
                 best_epoch: int = -1): 
        self.train_data = train_data
        self.val_data   = val_data
        self.encoder = encoder
        self.optimizer = optimizer 
        self.num_epochs = num_epochs
        self.num_blocks = num_blocks
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_models_to_keep = num_models_to_keep
        self.generate_after_n = generate_after_n
        self.best_epoch = best_epoch

        self.loss_fxn = torch.nn.CrossEntropyLoss()
        self.xent_loss_fxn = torch.nn.CrossEntropyLoss()
        self.nll_loss_fxn = torch.nn.NLLLoss()
        self.fore_loss_fxn = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.device = device
        self.compute_block_dist = self.encoder.compute_block_dist

    def train(self):
        all_accs = []
        max_acc = 0.0 
        for epoch in range(self.best_epoch + 1, self.num_epochs, 1): 
            acc, __ = self.train_and_validate_one_epoch(epoch)
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
        skipped = 0
        for b, batch_trajectory in tqdm(enumerate(self.train_data)): 
            #print(f"batch {b} has trajectory of length {len(batch_trajectory.to_iterate)}") 
            for i, batch_instance in enumerate(batch_trajectory): 
                #self.generate_debugging_image(batch_instance, f"input_batch_{b}_image_{i}_gold", is_input = True)
                self.optimizer.zero_grad() 
        
                outputs = self.encoder(batch_instance) 
                # skip bad examples 
                if outputs is None:
                    skipped += 1
                    continue
                loss = self.compute_loss(batch_instance, outputs) 
                #print(f"loss {loss.item()}") 
                loss.backward() 
                self.optimizer.step() 
        print(f"Validating epoch {epoch}...") 
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
        print(f"Epoch {epoch} has pixel acc {mean_acc * 100}, block acc {mean_block_acc * 100}") 
        # TODO (elias): change back to pixel acc after debugging 
        return mean_acc, mean_block_acc 

    def validate(self, batch_instance, epoch_num, batch_num, instance_num): 
        outputs = self.encoder(batch_instance) 
        accuracy = self.compute_localized_accuracy(batch_instance, outputs) 
        if self.compute_block_dist:
            block_accuracy = self.compute_block_accuracy(batch_instance, outputs) 
        else:
            block_accuracy = -1.0
            
        if epoch_num > self.generate_after_n: 
            for i in range(outputs["next_position"].shape[0]):
                output_path = self.checkpoint_dir.joinpath(f"batch_{batch_num}").joinpath(f"instance_{i}")
                output_path.mkdir(parents = True, exist_ok=True)
                self.generate_debugging_image(outputs["next_position"][i], output_path.joinpath("pred"))
                self.generate_debugging_image(batch_instance["next_position"][i], output_path.joinpath("gold"))

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

    def generate_debugging_image(self, data, out_path, is_input=False):
        # 9 x 21 x 4 x 64 x 64 
        if data.shape[0] == self.num_blocks + 1:
            # take argmax if distribution 
            data_id, data= torch.max(data, dim = 0) 
        data = data.squeeze(0)

        xs = np.arange(0, 64, 1)
        zs = np.arange(0, 64, 1)
        # separate plot per depth slice 
        # TODO (elias) make sure that heights are actually capped at 4
        for height in range(4):
            fig = plt.figure(figsize=(12,12))
            ax = fig.gca()
            ax.set_xticks([0, 16, 32, 48, 64])
            ax.set_yticks([0, 16, 32, 48, 64]) 
            ax.set_ylim(0, 64)
            ax.set_xlim(0, 64)
            plt.grid() 

            to_plot_xs, to_plot_zs, to_plot_labels = [], [], []
            for x_pos in xs:
                for z_pos in zs:
                    label = data[x_pos, z_pos, height].item() 
                    if height > 0 and label > 0:
                        pass 
                        #print(f"we have a tall block with label {label} at x, z, y: {x_pos, z_pos, height}")
                    # don't plot background 
                    if label != 0: 
                        label = int(label) 
                        to_plot_xs.append(x_pos)
                        to_plot_zs.append(z_pos)
                        to_plot_labels.append(label) 

            ax.plot(to_plot_xs, to_plot_zs, ".")
            for x,z, lab in zip(to_plot_xs, to_plot_zs, to_plot_labels):
                ax.annotate(lab, xy=(x,z), fontsize = 12)

            file_path =  f"{out_path}-{height}.png"
                
            print(f"saving to {file_path}") 
            plt.savefig(file_path) 
            plt.close() 

    def compute_block_accuracy(self, inputs, outputs): 
        pred_block_logits = outputs["pred_block_logits"] 
        true_block_idxs  = inputs["block_to_move"]
        true_block_idxs = true_block_idxs.to(self.device).long().reshape(-1) 
      
        pred_block_decisions = torch.argmax(pred_block_logits, dim = -1)
        num_correct = torch.sum(pred_block_decisions == true_block_idxs).detach().cpu().item() 
        accuracy = num_correct / true_block_idxs.shape[0]
        return accuracy

    def compute_loss(self, inputs, outputs):
        pred_image = outputs["next_position"]
        true_image = inputs["next_position"]
        prev_image  = inputs["previous_position_for_acc"]

        pred_block_logits = outputs["pred_block_logits"] 
        true_block_idxs = inputs["block_to_move"]
        true_block_idxs = true_block_idxs.to(self.device).long().reshape(-1) 

        bsz, n_blocks, width, height, depth = pred_image.shape
        true_image = true_image.reshape((bsz, width, height, depth)).long()
                
        true_image = true_image.to(self.device) 
        prev_image = prev_image.reshape((bsz, width, height, depth)).long()
        prev_image = prev_image.to(self.device) 
        
        if self.compute_block_dist:
            # loss per pixel 
            #pixel_loss = self.nll_loss_fxn(pred_image, true_image) 
            # TODO (elias): for now just do as auxiliary task 
            pixel_loss = self.xent_loss_fxn(pred_image, true_image) 
            foreground_loss = self.fore_loss_fxn(pred_image, true_image) 
            # loss per block
            block_loss = self.xent_loss_fxn(pred_block_logits, true_block_idxs) 
            #print(f"computing loss with blocks {pixel_loss.item()} + {block_loss.item()}") 
            total_loss = pixel_loss + block_loss + foreground_loss
            #total_loss = block_loss
        else:
            # loss per pixel 
            pixel_loss = self.xent_loss_fxn(pred_image, true_image) 
            # foreground loss 
            foreground_loss = self.fore_loss_fxn(pred_image, true_image) 
            #print(f"computing loss no blocks {pixel_loss.item()}") 
            total_loss = pixel_loss + foreground_loss

        print(f"loss {total_loss.item()}")
        return total_loss

    def compute_loss_no_blocks(self, inputs, outputs):
        pred_image = outputs["next_position"]
        true_image = inputs["next_position"]
        prev_image  = inputs["previous_position_for_acc"]

        bsz, n_blocks, width, height, depth = pred_image.shape
        true_image = true_image.reshape((bsz, width, height, depth)).long()
        true_image = true_image.to(self.device) 
        prev_image = prev_image.reshape((bsz, width, height, depth)).long()
        prev_image = prev_image.to(self.device) 
        
        # one total per-pixel loss 
        total_loss = self.loss_fxn(pred_image, true_image) 
        # one loss on only foreground pixels, excludes index 0 
        fore_loss = self.fore_loss_fxn(pred_image, true_image) 

        # find changes based on gold image 
        pixels_of_interest = true_image != prev_image
        # expand mask 
        pixels_of_interest_for_pred = pixels_of_interest.unsqueeze(1).repeat(1, n_blocks, 1, 1, 1)
        # isolate region 
        gold_pixels_of_interest = true_image[pixels_of_interest]
        pred_pixels_of_interest = pred_image[pixels_of_interest_for_pred].reshape((-1, n_blocks))

        # two losses for just region of interest 
        total_interest_loss = self.loss_fxn(pred_pixels_of_interest, gold_pixels_of_interest) 
        fore_interest_loss = self.fore_loss_fxn(pred_pixels_of_interest, gold_pixels_of_interest) 
        
        # downweight overall loss, upweight regional loss 
        #loss = 0.05 * (total_loss + fore_loss) +  (total_interest_loss + fore_interest_loss) 
        #loss = total_interest_loss
        loss = total_loss

        pred_image = torch.argmax(pred_image, dim = 1).squeeze(1)
        
        return loss 

    def save_model(self, epoch, is_best):
        print(f"Saving checkpoint {epoch}") 
        # get path 
        save_path = self.checkpoint_dir.joinpath(f"model_{epoch}.th") 
        torch.save(self.encoder.state_dict(), save_path) 
        print(f"Saved checkpoint to {save_path}") 
        # if it's best performance, save extra 
        if is_best:
            best_path = self.checkpoint_dir.joinpath(f"best.th") 
            torch.save(self.encoder.state_dict(), best_path) 

            json_info = {"epoch": epoch}
            with open(self.checkpoint_dir.joinpath("best_training_state.json"), "w") as f1:
                json.dump(json_info, f1) 

            print(f"Updated best model to {best_path} at epoch {epoch}") 

        # remove old models 
        all_paths = list(self.checkpoint_dir.glob("model_*th"))
        if len(all_paths) > self.num_models_to_keep:
            to_remove = sorted(all_paths, key = lambda x: int(os.path.basename(x).split(".")[0].split('_')[1]))[0:-self.num_models_to_keep]
            for path in to_remove:
                os.remove(path) 

    def evaluate(self):
        total_acc = 0.0 
        total = 0 
        total_block_acc = 0.0 
        self.encoder.eval() 
        for b, dev_batch_trajectory in tqdm(enumerate(self.val_data)): 
            for i, dev_batch_instance in enumerate(dev_batch_trajectory): 
                pixel_acc, block_acc = self.validate(dev_batch_instance, 1, b, i) 
                total_acc += pixel_acc
                total_block_acc += block_acc
                total += 1

        mean_acc = total_acc / total 
        mean_block_acc = total_block_acc / total
        print(f"Test-time pixel acc {mean_acc * 100}, block acc {mean_block_acc * 100}") 
        return mean_acc 

class FlatLanguageTrainer(LanguageTrainer):
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
                 best_epoch: int = -1): 
        super(FlatLanguageTrainer, self).__init__(train_data,
                                                  val_data,
                                                  encoder,
                                                  optimizer,
                                                  num_epochs,
                                                  num_blocks,
                                                  device,
                                                  checkpoint_dir,
                                                  num_models_to_keep,
                                                  generate_after_n,
                                                  best_epoch)

    def train_and_validate_one_epoch(self, epoch): 
        print(f"Training epoch {epoch}...") 
        self.encoder.train() 
        skipped = 0
        for b, batch_instance in tqdm(enumerate(self.train_data)): 
            self.optimizer.zero_grad() 
            outputs = self.encoder(batch_instance) 
            # skip bad examples 
            if outputs is None:
                skipped += 1
                continue
            loss = self.compute_loss(batch_instance, outputs) 
            loss.backward() 
            self.optimizer.step() 

        print(f"skipped {skipped} examples") 
        print(f"Validating epoch {epoch}...") 
        total_acc = 0.0 
        total = 0 
        total_block_acc = 0.0 

        self.encoder.eval() 
        for b, dev_batch_instance in tqdm(enumerate(self.val_data)): 
            pixel_acc, block_acc = self.validate(dev_batch_instance, epoch, b, 0) 
            total_acc += pixel_acc
            total_block_acc += block_acc
            total += 1

        mean_acc = total_acc / total 
        mean_block_acc = total_block_acc / total
        print(f"Epoch {epoch} has pixel acc {mean_acc * 100}, block acc {mean_block_acc * 100}") 
        # TODO (elias): change back to pixel acc after debugging 
        return mean_acc, mean_block_acc 

    def evaluate(self):
        total_acc = 0.0 
        total = 0 
        total_block_acc = 0.0 
        self.encoder.eval() 
        for b, dev_batch_instance in tqdm(enumerate(self.val_data)): 
            pixel_acc, block_acc = self.validate(dev_batch_instance, 1, b, 0) 
            total_acc += pixel_acc
            total_block_acc += block_acc
            total += 1

        mean_acc = total_acc / total 
        mean_block_acc = total_block_acc / total
        print(f"Test-time pixel acc {mean_acc * 100}, block acc {mean_block_acc * 100}") 
        return mean_acc 


def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]).decode("utf-8") 
    gpu_df = pd.read_csv(StringIO(u"".join(gpu_stats)),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
    gpu_df['memory.used'] = gpu_df['memory.used'].map(lambda x: x.rstrip(' [MiB]'))
    gpu_df['memory.free'] = gpu_df['memory.free'].astype(np.int64)
    gpu_df['memory.used'] = gpu_df['memory.used'].astype(np.int64)
    idx = gpu_df['memory.free'].idxmax()
    if gpu_df["memory.used"][idx] > 0.0:
        print(f"No free gpus!") 
        return -1
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx



def main(args):
    # load the data 
    dataset_reader = DatasetReader(args.train_path,
                                   args.val_path,
                                   None,
                                   batch_by_line = args.traj_type != "flat",
                                   traj_type = args.traj_type,
                                   batch_size = args.batch_size,
                                   max_seq_length = args.max_seq_length)  

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

    device = "cpu"
    if args.cuda is not None:
        free_gpu_id = get_free_gpu()
        if free_gpu_id > -1:
            device = f"cuda:{free_gpu_id}"

    device = torch.device(device)  
    print(f"On device {device}") 

    # construct image encoder 
    flatten = args.fuser == "concat" 
    image_encoder = ImageEncoder(input_dim = 2, 
                                 n_layers = args.conv_num_layers,
                                 factor = args.conv_factor,
                                 dropout = args.dropout,
                                 flatten = flatten)

    # construct image and language fusion module 
    fusion_options = {"concat": ConcatFusionModule,
                      "tiled": TiledFusionModule}

    encoder_hidden_dim = encoder.hidden_dim
    if encoder.bidirectional:
        encoder_hidden_dim *= 2 

    fuser = fusion_options[args.fuser](image_encoder.output_dim, encoder_hidden_dim) 

    # construct image decoder 
    deconv_options = {"coupled": DeconvolutionalNetwork,
                      "decoupled": DecoupledDeconvolutionalNetwork}

    output_module = deconv_options[args.deconv](input_channels  = fuser.output_dim, 
                                   num_blocks = args.num_blocks,
                                   num_layers = args.deconv_num_layers,
                                   dropout = args.dropout,
                                   flatten = flatten,
                                   factor = args.deconv_factor,
                                   initial_width = 6) 

    block_prediction_module = MLP(input_dim = fuser.output_dim,
                                  hidden_dim = args.mlp_hidden_dim, 
                                  output_dim = args.num_blocks+1, 
                                  num_layers = args.mlp_num_layers, 
                                  dropout = args.mlp_dropout)  

    # put it all together into one module 
    encoder = LanguageEncoder(image_encoder = image_encoder, 
                              embedder = embedder, 
                              encoder = encoder, 
                              fuser = fuser, 
                              output_module = output_module,
                              block_prediction_module = block_prediction_module,
                              device = device,
                              compute_block_dist = args.compute_block_dist) 

    # construct optimizer 
    optimizer = torch.optim.Adam(encoder.parameters())

    if args.traj_type == "flat":
        trainer_cls = FlatLanguageTrainer
    else:
        trainer_cls = LanguageTrainer

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
        trainer = trainer_cls(train_data = dataset_reader.data["train"], 
                              val_data = dataset_reader.data["dev"], 
                              encoder = encoder,
                              optimizer = optimizer, 
                              num_epochs = args.num_epochs,
                              num_blocks = args.num_blocks,
                              device = device,
                              checkpoint_dir = args.checkpoint_dir,
                              num_models_to_keep = args.num_models_to_keep,
                              generate_after_n = args.generate_after_n, 
                              best_epoch = best_epoch) 
        print(encoder)
        trainer.train() 

    else:
        # test-time, load best model  
        print(f"loading model weights from {args.checkpoint_dir}") 
        state_dict = torch.load(pathlib.Path(args.checkpoint_dir).joinpath("best.th"))
        encoder.load_state_dict(state_dict, strict=True)  

        eval_trainer = trainer_cls(train_data = dataset_reader.data["train"], 
                                           val_data = dataset_reader.data["dev"], 
                                           encoder = encoder,
                                           optimizer = None, 
                                           num_epochs = 0, 
                                           device = device,
                                           checkpoint_dir = args.checkpoint_dir,
                                           num_models_to_keep = 0, 
                                           generate_after_n = 0) 
        print(f"evaluating") 
        eval_trainer.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="load model and test")
    parser.add_argument("--resume", action="store_true", help="resume training a model")
    # data 
    parser.add_argument("--train-path", type=str, default = "blocks_data/trainset_v2.json", help="path to train data")
    parser.add_argument("--val-path", default = "blocks_data/devset.json", type=str, help = "path to dev data" )
    parser.add_argument("--num-blocks", type=int, default=20) 
    parser.add_argument("--traj-type", type=str, default="flat", choices = ["flat", "trajectory"]) 
    parser.add_argument("--batch-size", type=int, default = 32) 
    parser.add_argument("--max-seq-length", type=int, default = 65) 
    parser.add_argument("--do-filter", action="store_true", help="set if we want to restrict prediction to the block moved") 
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
    parser.add_argument("--deconv", type=str, default="coupled", choices=["coupled", "decoupled"]) 
    parser.add_argument("--deconv-factor", type=int, default = 2) 
    parser.add_argument("--deconv-num-layers", type=int, default=2) 
    # fuser
    parser.add_argument("--fuser", type=str, default="concat", choices=["tiled", "concat"]) 
    # block mlp
    parser.add_argument("--compute-block-dist", action="store_true") 
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

