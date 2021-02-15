import torch
import numpy as np
from spacy.tokenizer import Tokenizer 
from transformers import BertTokenizer, BertModel
import pdb

np.random.seed(12) 
torch.manual_seed(12) 

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
        self.output_dim = embedding_dim 

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
        self.output_dim = embedding_dim 

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

class BERTEmbedder(torch.nn.Module): 
    def __init__(self, 
                 model_name: str = "bert-base-uncased", 
                 max_seq_len: int = 60, 
                 trainable: bool = False): 
        super(BERTEmbedder, self).__init__()

        self.model_name = model_name 
        self.trainable = trainable
        self.max_seq_len = max_seq_len

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name) 
        self.bert_model = BertModel.from_pretrained(self.model_name) 
        self.output_dim = 768
        self.bert_model.eval() 

    def set_device(self, device):
        self.device = device
        if "cuda" in str(device):
            self.bert_model = self.bert_model.to(device) 

    def forward(self, words):
        words = [x if x != "<PAD>" else "[PAD]" for x  in words]
        text = " ".join(words)
        tokenized_text = self.tokenizer.tokenize(text)[0:self.max_seq_len ]
        if len(tokenized_text) < self.max_seq_len:
            # happens when there is weird whitepsace in the command 
            pads = ["[PAD]" for i in range(self.max_seq_len - len(tokenized_text))]
            tokenized_text += pads 

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device) 
        
        if not self.trainable: 
            with torch.no_grad():
                outputs = self.bert_model(tokens_tensor) 
        else:
            outputs = self.bert_model(tokens_tensor, token_type_ids=segments_tensors)

        # use top layer 
        encoded_sequence = outputs[0]

        return encoded_sequence.squeeze(0) 
