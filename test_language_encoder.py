import json 
import argparse
from typing import List, Dict

import torch
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from language import LanguageEncoder
from encoders import LSTMEncoder
from language_embedders import RandomEmbedder


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
                 encoder: LanguageEncoder):
        self.train_data = train_data
        self.val_data   = val_data
        self.encoder = encoder
        # TODO: get a loss here  
        # self.loss = torch.nn.CrossEn


    def train_epoch(self): 
        for row in self.train_data:
            for j, sent in enumerate(row["notes"]): 
                sent = [sent["notes"][0]]
                output = self.encoder(sent) 

    def validate(self): 
        pass 


def main(args):
    # load the data 
    train_data = load_data(args.train_path) 
    val_data = load_data(args.val_path) 
    # construct the vocab 
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)
    vocab = get_vocab(train_data, tokenizer)     
    
    # get the embedder from args 
    if args.embedder == "random":
        embedder = RandomEmbedder(tokenizer, vocab, args.embedding_dim, trainable=True)
    else:
        raise NotImplementedError(f"No embedder {args.embedder}") 
    # get the encoder from args  
    if args.encoder == "lstm":
        encoder = LSTMEncoder(input_dim = args.embedding_dim,
                              hidden_dim = args.hidden_dim,
                              num_layers = args.num_layers,
                              dropout = args.dropout,
                              bidirectional = args.bidirectional) 
    else:
        raise NotImplementedError(f"No encoder {args.encoder}") 
    # construct the model 
    encoder = LanguageEncoder(embedder, encoder, output_type=args.output_type) 
    trainer = LanguageTrainer(train_data, val_data, encoder) 
    trainer.train_epoch() 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str)
    parser.add_argument("--val-path", type=str)
    parser.add_argument("--embedder", type=str, default="random")
    parser.add_argument("--embedding-dim", type=int, default=300) 
    parser.add_argument("--encoder", type=str, default="lstm")
    parser.add_argument("--output-type", type=str, default="mask")
    parser.add_argument("--hidden-dim", type=int, default=128) 
    parser.add_argument("--num-layers", type=int, default=2) 
    parser.add_argument("--dropout", type=float, default=0.2) 
    parser.add_argument("--bidirectional", action="store_true") 

    args = parser.parse_args()
    main(args) 

