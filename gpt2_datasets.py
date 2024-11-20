from torch.utils.data import Dataset
from modules.utils import loadpkl
import os
import torch
from transformers import GPT2Tokenizer
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STdata(Dataset):
    def __init__(self, name, dataroot="data", max_len=15):
        super(STdata, self).__init__()
        assert name in ['train', 'valid', 'test']
        self.name = name
        self.word2idx = loadpkl(dataroot+'/word2idx.pkl')
        self.idx2word = loadpkl(dataroot+'/idx2word.pkl')

        self.vocab_size = len(self.word2idx)

        input_path = os.path.join(dataroot, name+'/src.pkl')
        output_path = os.path.join(dataroot, name+'/trg.pkl')
        sim_path = os.path.join(dataroot, name+'/sim.pkl')

        gpt_input_path = os.path.join(dataroot, name+'/gpt_src.pkl')
        gpt_output_path = os.path.join(dataroot, name+'/gpt_trg.pkl')

        self.input_s = loadpkl(input_path)
        self.output_s = loadpkl(output_path)
        self.sim_s = loadpkl(sim_path)
        self.gpt_input = loadpkl(gpt_input_path)
        self.gpt_output = loadpkl(gpt_output_path)

        self.max_len = max_len

        # Load GPT-2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set EOS as padding token for GPT-2

    def __getitem__(self, index):
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
            trg[self.max_len] = self.tokenizer.eos_token_id  # EOS token
            trg_input[1:self.max_len + 1] = torch.tensor(output_s[0:self.max_len])
            trg_input[0] = self.tokenizer.bos_token_id  # BOS token for GPT-2
            ou_len = self.max_len + 1
        else:
            trg[0:ou_len] = torch.tensor(output_s)
            trg[ou_len] = self.tokenizer.eos_token_id  # EOS token
            trg_input[1:ou_len + 1] = torch.tensor(output_s)
            trg_input[0] = self.tokenizer.bos_token_id  # BOS token for GPT-2
            ou_len = ou_len + 1

        # GPT-2 data format (we no longer need [CLS] or [SEP] tokens)
        gpt_in = self.gpt_input[index]
        gpt_out = self.gpt_output[index]
        sim = self.gpt_output[random.choice(self.sim_s[index])]
        gpt_src = torch.zeros(self.max_len + 2, dtype=torch.long)
        gpt_trg = torch.zeros(self.max_len + 2, dtype=torch.long)
        gpt_sim = torch.zeros(self.max_len + 2, dtype=torch.long)

        gpt_src[0:min(len(gpt_in), self.max_len + 2)] = torch.tensor(gpt_in[0:min(self.max_len + 2, len(gpt_in))])
        gpt_trg[0:min(len(gpt_out), self.max_len + 2)] = torch.tensor(gpt_out[0:min(self.max_len + 2, len(gpt_out))])
        gpt_sim[0:min(len(sim), self.max_len + 2)] = torch.tensor(sim[0:min(self.max_len + 2, len(sim))])

        return src, in_len, trg, trg_input, ou_len, gpt_src, gpt_trg, gpt_sim, content_trg, content_len

    def __len__(self):
        return len(self.input_s)
