import json 
import argparse
from typing import List, Dict

import torch
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import logging 
from tqdm import tqdm 

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
                 num_epochs: int): 
        self.train_data = train_data
        self.val_data   = val_data
        self.encoder = encoder
        # TODO: get a loss here  
        # self.loss = torch.nn.CrossEn
        self.optimizer = optimizer 
        self.num_epochs = num_epochs

        self.loss_fxn = torch.nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(self.num_epochs): 
            self.train_epoch(epoch)

    def train_epoch(self, epoch): 
        print(f"Training epoch {epoch}...") 
        for batch_trajectory in tqdm(self.train_data): 
            for batch_instance in batch_trajectory: 
                self.optimizer.zero_grad() 
                outputs = self.encoder(batch_instance) 
                loss = self.compute_loss(batch_instance, outputs) 
                loss.backward() 
                print(loss.item() ) 
                self.optimizer.step() 

        print(f"Validating epoch {epoch}...") 
        with self.encoder.eval(): 
            for dev_batch_instance in tqdm(self.val_data): 
                val_outputs = self.validate(dev_batch_instance) 


    def validate(self, batch_instance): 
        return None 

    def compute_loss(self, inputs, outputs):
        pred_image = outputs["next_position"]
        true_image = inputs["next_position"]

        bsz, n_blocks, width, height, depth = pred_image.shape
        true_image = true_image.reshape((bsz, width, height, depth)).long()

        loss = self.loss_fxn(pred_image, true_image) 

        return loss 


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

    # put it all together into one module 
    encoder = LanguageEncoder(image_encoder = image_encoder, 
                              embedder = embedder, 
                              encoder = encoder, 
                              fuser = fuser, 
                              output_module = output_module) 
    # construct optimizer 
    optimizer = torch.optim.Adam(encoder.parameters())
    # construct trainer 
    trainer = LanguageTrainer(train_data = dataset_reader.data["train"], 
                              val_data = dataset_reader.data["dev"], 
                              encoder = encoder,
                              optimizer = optimizer, 
                              num_epochs = 3) 
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

    args = parser.parse_args()
    main(args) 

