import torch
from torch.utils.data import Dataset
import os
from transformers import RobertaTokenizer
import random
import pickle

def loadpkl(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
    return obj

class STdata(Dataset):
    def __init__(self, name, dataroot="data", max_len=15):
        super(STdata, self).__init__()
        assert name in ['train', 'valid', 'test']
        self.name = name
        self.word2idx = loadpkl(dataroot + '/word2idx.pkl')
        self.idx2word = loadpkl(dataroot + '/idx2word.pkl')

        self.vocab_size = len(self.word2idx)

        input_path = os.path.join(dataroot, name + '/src.pkl')
        output_path = os.path.join(dataroot, name + '/trg.pkl')
        sim_path = os.path.join(dataroot, name + '/sim.pkl')

        roberta_input_path = os.path.join(dataroot, name + '/roberta_src.pkl')
        roberta_output_path = os.path.join(dataroot, name + '/roberta_trg.pkl')

        self.input_s = loadpkl(input_path)
        self.output_s = loadpkl(output_path)
        self.sim_s = loadpkl(sim_path)
        self.roberta_input = loadpkl(roberta_input_path)
        self.roberta_output = loadpkl(roberta_output_path)

        self.max_len = max_len

    def __getitem__(self, index):
        # seq2seq data_quora format
        input_s = self.input_s[index]
        output_s = self.output_s[index]

        in_len = len(input_s)
        ou_len = len(output_s)

        src = torch.zeros(self.max_len, dtype=torch.long)
        content_trg = torch.zeros(self.max_len, dtype=torch.long)
        content_len = 0

        trg = torch.zeros(self.max_len + 1, dtype=torch.long)
        trg_input = torch.zeros(self.max_len + 1, dtype=torch.long)

        if in_len > self.max_len:
            src[0:self.max_len] = torch.tensor(input_s[0:self.max_len])
            in_len = self.max_len
        else:
            src[0:in_len] = torch.tensor(input_s)

        if ou_len > self.max_len:
            content_trg[0:self.max_len] = torch.tensor(output_s[0:self.max_len])
            content_len = self.max_len
        else:
            content_trg[0:ou_len] = torch.tensor(output_s)
            content_len = ou_len

        if ou_len > self.max_len:
            trg[0:self.max_len] = torch.tensor(output_s[0:self.max_len])
            trg[self.max_len] = 2  # EOS token for RoBERTa
            trg_input[1:self.max_len + 1] = torch.tensor(output_s[0:self.max_len])
            trg_input[0] = 0  # Start of sequence token for RoBERTa
            ou_len = self.max_len + 1
        else:
            trg[0:ou_len] = torch.tensor(output_s)
            trg[ou_len] = 2  # EOS
            trg_input[1:ou_len + 1] = torch.tensor(output_s)
            trg_input[0] = 0  # Start of sequence
            ou_len = ou_len + 1

        # RoBERTa data_quora format
        roberta_in = self.roberta_input[index]
        roberta_out = self.roberta_output[index]
        sim = self.roberta_output[random.choice(self.sim_s[index])]
        roberta_src = torch.zeros(self.max_len + 2, dtype=torch.long)
        roberta_trg = torch.zeros(self.max_len + 2, dtype=torch.long)
        roberta_sim = torch.zeros(self.max_len + 2, dtype=torch.long)

        roberta_src[0:min(len(roberta_in), self.max_len + 2)] = torch.tensor(roberta_in[0:min(self.max_len + 2, len(roberta_in))])
        roberta_trg[0:min(len(roberta_out), self.max_len + 2)] = torch.tensor(roberta_out[0:min(self.max_len + 2, len(roberta_out))])
        roberta_sim[0:min(len(sim), self.max_len + 2)] = torch.tensor(sim[0:min(self.max_len + 2, len(sim))])

        return src, in_len, trg, trg_input, ou_len, roberta_src, roberta_trg, roberta_sim, content_trg, content_len

    def __len__(self):
        return len(self.input_s)
