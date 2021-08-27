import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import config
import deprecated.config
from learning.modules.cuda_module import CudaModule as ModuleBase
from learning.inputs.sequence import sequence_list_to_tensor
from learning.inputs.common import empty_float_tensor
from data_io.paths import get_self_attention_path
import pickle
import matplotlib.pyplot as plt
import numpy as np

class SentenceEmbeddingSelfAttentionCond(ModuleBase):

    def __init__(self,
                 word_embedding_size,
                 lstm_size,
                 lstm_layers=1,
                 attention_heads=5,
                 run_name="", BiLSTM=True,
                 dropout = False,
                 hc = 32,
                 k = 5,
                 stride = 2):

        pad = int(k/2)
        super(SentenceEmbeddingSelfAttentionCond, self).__init__()
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.embedding = nn.Embedding(deprecated.config.vocab_size, word_embedding_size, sparse=False)
        self.BiLSTM = BiLSTM
        self.dropout = dropout
        self.factor = (2 if self.BiLSTM else 1)
        self.lstm_txt = nn.LSTM(word_embedding_size, self.lstm_size, self.lstm_layers, bidirectional=BiLSTM)
        self.Da = 25
        # TODO: Make sure the overall embedding is of the size requested
        self.num_attn_heads = attention_heads
        self.W_s1 = nn.Linear(self.factor * self.lstm_size, self.Da, bias = False)
        self.W_s2 = nn.Linear(self.Da, self.num_attn_heads, bias = False)

        self.idx2word = pickle.load(open(get_self_attention_path()+"idx2word.pickle", "rb"))
        self.n_epoch = 0
        self.n_batch = 0

        #conv for feature map
        self.conv1 = nn.Conv2d(hc, hc, k, stride=stride, padding=pad)
        self.conv2 = nn.Conv2d(hc, hc, k, stride=stride, padding=pad)
        self.Linear_FeatureMap = nn.Linear(32*8*8, self.Da)

        self.dropout2d = nn.Dropout2d(0.6)

        self.init_weights()


    def init_weights(self):
        self.embedding.weight.data.normal_(0, 1)
        #self.embedding.weight.data.uniform_(-0.1, 0.1)
        for name, param in self.lstm_txt.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv3.weight)
        self.conv3.bias.data.fill_(0)

    def save_att_map(self, batch_num, epoch_num):
        self.n_batch = batch_num
        self.n_epoch = epoch_num
        return

    def forward(self, word_ids, feature_map, lengths=None):
        # TODO: Get rid of this and abstract in another layer

        if isinstance(word_ids, list) and lengths is None:
            word_ids, lengths = sequence_list_to_tensor([word_ids])
            word_ids = word_ids.to(feature_map.device)
            lengths = lengths.to(feature_map.device)

        word_embeddings = self.embedding(word_ids) #size: [2, 500, 20] embedding size: 20
        batch_size = word_embeddings.size(0) # size:2
        sentence_embeddings = Variable(empty_float_tensor((batch_size, self.lstm_size*self.factor*(self.num_attn_heads+1)), self.is_cuda, self.cuda_device)) #size [2,80]

        penal = 0

        for i in range(batch_size):
            length = int(lengths[i])
            if length == 0:
                print("Empty caption")
                continue
            embeddings_i = word_embeddings[i, 0:length].unsqueeze(1) # size: [instruction length, 1, 20]
            h0 = Variable(empty_float_tensor((self.lstm_layers*self.factor, 1, self.lstm_size), self.is_cuda)) #size: [2, 1, 40]
            c0 = Variable(empty_float_tensor((self.lstm_layers*self.factor, 1, self.lstm_size), self.is_cuda)) #size: [2, 1, 40]
            outputs, states = self.lstm_txt(embeddings_i, (h0, c0)) #output size: [intr_len, 1, 80]  #2 states: forward and backwward.  size: [2, 1, 40]
            H = outputs.squeeze(dim=1) #size: [instr_len, 80]
            hidden, cell = (states[0].squeeze(dim=1), states[1].squeeze(dim=1)) #size: 2x[2,40]

            #image key
            # TODO: This is one good option (but leakyReLU)
            k1 = F.leaky_relu(self.conv1(feature_map)) #
            k1_dropout = self.dropout2d(k1)
            k2 = F.leaky_relu(self.conv2(k1_dropout)) #
            k2_drop = self.dropout2d(k2)
            key = self.Linear_FeatureMap(k2_drop.view(-1))
            # TODO: Dropout

            # TODO: Alternative: pooling

            #self-attention
            s1 = F.tanh(self.W_s1(H))
            s2_fixed = self.W_s2(s1)
            s2_dynamic = torch.mm(s1, key.view(-1,1))
            s2_cat = torch.cat((s2_fixed, s2_dynamic), dim=1)

            A = F.softmax(s2_cat.t(), dim=1)
            M = torch.mm(A, H)

            # if self.is_cuda:
            #     I = Variable(torch.eye(self.num_attn_heads).cuda())
            # else:
            #     I = Variable(torch.eye(self.num_attn_heads))

            AAt = torch.mm(A, A.t())
            for j in range(self.num_attn_heads):
                AAt[j, j] = 0
            p = torch.norm(AAt, 2)
            penal += p*p


        penal /= batch_size
        # Mean-reduce the 1st (sequence) dimension
        #sentence_embedding = torch.mean(M, 0) #size [80]
        sentence_embedding = M.view(-1)
        sentence_embeddings[i] = sentence_embedding.squeeze()

        if self.n_batch%2000 == 0:
            str_id = word_ids[-1][:length].data.cpu().numpy()
            instr = [self.idx2word[str(i)] for i in str_id]
            Att = A.data.cpu().numpy()
            filepath = get_self_attention_path() + "sample_instructions/sample_intr-{}-{}.txt".format(self.n_epoch, self.n_batch)
            # with open(filepath, "w") as f:
            #     for w in zip(instr, Att[0], Att[1], Att[2], Att[3], Att[4]):
            #         f.write(str(w)+"\n")

            imgpath = get_self_attention_path() + "instruction_heatmap/intr_heatmap-{}-{}.png".format(self.n_epoch, self.n_batch)

            # plt.close()
            plt.figure(figsize=(len(instr) / 6, 1.8))
            plt.pcolor(Att)
            plt.xticks(np.linspace(0.5, len(instr)-0.5, len(instr)), instr, rotation=90, fontsize=10)
            plt.gcf().subplots_adjust(bottom=0.5)
            plt.savefig(imgpath)
            # plt.show()
            self.n_batch += 1

        return sentence_embeddings, penal