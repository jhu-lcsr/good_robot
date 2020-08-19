import torch
import numpy as np
from spacy.tokenizer import Tokenizer 

class GloveEmbedder(torch.nn.Module):
    def __init__(self, 
                 tokenizer: Tokenizer,
                 embedding_file: str,
                 trainable: bool = False): 
        super(GloveEmbedder, self).__init__()

        self.tokenizer = tokenizer
        self.trainable = trainable
        
        # read embeddings from file 
        with open(embedding_file) as f1:
            contents = f1.readlines()

        embedding_dict = {}
        for line in contents: 
            line = line.split(" ")
            key = line[0]
            embeddings = [float(x) for x in line[1:]]
            assert(len(embeddings) == 300)
            embedding_dict[key] = np.array(embeddings)
        
        # set embeddings 
        self.embedding_dict = embedding_dict
        self.unk_embedding = np.zeros(300)
        self.vocab = self.embedding_dict.keys() 

    def forward(self, words):
        words = self.tokenizer(words)
        if not self.trainable:
            with torch.no_grad():  
                output = [self.embedding_dict[w] if w in self.vocab else self.unk_embedding for w in words ]
                return torch.Tensor(output) 
        else:
            output = [self.embedding_dict[w] if w in self.vocab else self.unk_embedding for w in words ]
            return torch.Parameter(torch.Tensor(output))

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
        self.embedding_dict = {}
        self.unk_embedding = torch.zeros((1, embedding_dim))
        self.pad_token = torch.ones((1, embedding_dim))
        self.vocab = vocab

        for word in vocab:
            # TODO: give initalization 
            embedding = torch.nn.Parameter(torch.zeros((1, embedding_dim)))
            torch.nn.init.uniform_(embedding) 
            self.embedding_dict[word] = embedding

    def forward(self, words):
        output = [self.embedding_dict[w] if w in self.vocab else self.unk_embedding for w in words ]
        return torch.cat(output, dim = 0) 
