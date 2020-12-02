import torch
import numpy as np
from spacy.tokenizer import Tokenizer 
import pdb 

class GloveEmbedder(torch.nn.Module):
    def __init__(self, 
                 tokenizer: Tokenizer,
                 vocab: set,
                 embedding_file: str, 
                 embedding_dim: int = 300,
                 trainable: bool = True): 
        super(GloveEmbedder, self).__init__()

        self.tokenizer = tokenizer
        self.trainable = trainable
        
        # read embeddings from file 
        print(f"reading embeddings...") 
        with open(embedding_file) as f1:
            contents = f1.readlines()

        # set embeddings 
        self.unk_embedding = torch.zeros((1, embedding_dim))
        self.pad_token = torch.ones((1, embedding_dim))
        self.vocab = vocab
        self.word_to_idx = {word:i+2 for i, word in enumerate(vocab)}
        self.word_to_idx["<UNK>"] = 0
        self.word_to_idx["<PAD>"] = 1

        self.embeddings = torch.nn.Embedding(len(self.vocab) + 2, embedding_dim)

        fake_weight = torch.clone(self.embeddings.weight)

        for line in contents: 
            line = line.split(" ")
            key = line[0]
            embeddings = [float(x) for x in line[1:]]
            assert(len(embeddings) == embedding_dim)

            if key in self.vocab:
                # initialize with glove embedding when you can, otherwise keep random for unks 
                key_idx = self.word_to_idx[key]
                fake_weight[key_idx] = torch.tensor(np.array(embeddings)) 

        self.embeddings.load_state_dict({"weight": fake_weight}) 
        if not trainable:
            self.embeddings.weight.requires_grad = False
        else:
            self.embeddings.weight.requires_grad = True

    def set_device(self, device):
        self.device = device
        if "cuda" in str(device):
            self.embeddings = self.embeddings.cuda(self.device) 

    def forward(self, words):
        words = [w if w in self.vocab else "<UNK>" for w in words]
        lookup_tensor = torch.tensor([self.word_to_idx[w] for w in words], dtype = torch.long)
        lookup_tensor = lookup_tensor.to(self.device)
        output = self.embeddings(lookup_tensor)
        return output 

class RandomEmbedder(torch.nn.Module):
    def __init__(self, 
                 tokenizer: Tokenizer,
                 vocab: set,
                 embedding_dim: int,
                 trainable: bool = True): 
        super(RandomEmbedder, self).__init__() 

        self.tokenizer = tokenizer
        self.trainable = trainable

        # set embeddings 
        self.unk_embedding = torch.zeros((1, embedding_dim))
        self.pad_token = torch.ones((1, embedding_dim))
        self.vocab = vocab
        self.word_to_idx = {word:i+2 for i, word in enumerate(vocab)}
        self.word_to_idx["<UNK>"] = 0
        self.word_to_idx["<PAD>"] = 1

        self.embeddings = torch.nn.Embedding(len(self.vocab) + 2, embedding_dim)

    def set_device(self, device):
        self.device = device
        if "cuda" in str(device):
            self.embeddings = self.embeddings.cuda(self.device) 

    def forward(self, words):
        words = [w if w in self.vocab else "<UNK>" for w in words]
        lookup_tensor = torch.tensor([self.word_to_idx[w] for w in words], dtype = torch.long)
        lookup_tensor = lookup_tensor.to(self.device)
        output = self.embeddings(lookup_tensor)
        return output 










