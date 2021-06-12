import torch.nn as nn
from torch.autograd import Variable

import config
import deprecated.config
from learning.modules.module_base import ModuleBase
from learning.inputs.sequence import sequence_list_to_tensor
from learning.inputs.common import empty_float_tensor


class SentenceEmbeddingSimple(ModuleBase):

    def __init__(self, word_embedding_size, lstm_size, lstm_layers=1, run_name=""):
        super(SentenceEmbeddingSimple, self).__init__()
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.embedding = nn.Embedding(deprecated.config.vocab_size, word_embedding_size, sparse=False)
        self.lstm_txt = nn.LSTM(word_embedding_size, lstm_size, lstm_layers)

    def init_weights(self):
        self.embedding.weight.data.normal_(0, 1)
        #self.embedding.weight.data.uniform_(-0.1, 0.1)
        for name, param in self.lstm_txt.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

    def forward(self, word_ids, lengths=None):
        # TODO: Get rid of this and abstract in another layer
        if isinstance(word_ids, list) and lengths is None:
            word_ids, lengths = sequence_list_to_tensor([word_ids])
            if self.is_cuda:
                word_ids = word_ids.cuda()
                lengths = lengths.cuda()
        word_embeddings = self.embedding(word_ids)
        batch_size = word_embeddings.size(0)

        sentence_embeddings = Variable(empty_float_tensor((batch_size, self.lstm_size), self.is_cuda, self.cuda_device))

        for i in range(batch_size):
            length = int(lengths[i])
            if length == 0:
                #print("Empty caption")
                continue
            embeddings_i = word_embeddings[i, 0:length].unsqueeze(1)
            h0 = Variable(empty_float_tensor((self.lstm_layers, 1, self.lstm_size), self.is_cuda))
            c0 = Variable(empty_float_tensor((self.lstm_layers, 1, self.lstm_size), self.is_cuda))
            outputs, states = self.lstm_txt(embeddings_i, (h0, c0))
            # Mean-reduce the 1st (sequence) dimension
            sentence_embedding = outputs[-1]#torch.mean(outputs, 0)
            sentence_embeddings[i] = sentence_embedding.squeeze()

        return sentence_embeddings