from torch.utils.data import Dataset, DataLoader
import os
import pickle
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loadpkl(path):
    with open(path, 'rb') as f:
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

        bert_input_path = os.path.join(dataroot, name + '/bert_src.pkl')
        bert_output_path = os.path.join(dataroot, name + '/bert_trg.pkl')

        self.input_s = loadpkl(input_path)
        self.output_s = loadpkl(output_path)
        self.sim_s = loadpkl(sim_path)
        self.bert_input = loadpkl(bert_input_path)
        self.bert_output = loadpkl(bert_output_path)

        self.max_len = max_len

    def __getitem__(self, index):
        # Simply return raw data for processing in collate_fn
        return {
            "input_s": self.input_s[index],
            "output_s": self.output_s[index],
            "sim_s": self.sim_s[index],
            "bert_input": self.bert_input[index],
            "bert_output": self.bert_output[index],
            "global_bert_output": self.bert_output
        }

    def __len__(self):
        return len(self.input_s)


def collate_fn(batch, max_len=15):
    # Collate function to process the raw data in batches
    batch_size = len(batch)

    # Initialize tensors
    src = torch.zeros(batch_size, max_len, dtype=torch.long)
    trg = torch.zeros(batch_size, max_len + 1, dtype=torch.long)
    trg_input = torch.zeros(batch_size, max_len + 1, dtype=torch.long)
    content_trg = torch.zeros(batch_size, max_len, dtype=torch.long)
    bert_src = torch.zeros(batch_size, max_len + 2, dtype=torch.long)
    bert_trg = torch.zeros(batch_size, max_len + 2, dtype=torch.long)
    bert_sim = torch.zeros(batch_size, max_len + 2, dtype=torch.long)
    global_bert_output = batch[0]["global_bert_output"]

    in_len = []
    out_len = []
    content_len = []

    for i, sample in enumerate(batch):
        input_s = sample["input_s"]
        output_s = sample["output_s"]
        sim = sample["sim_s"]
        bert_in = sample["bert_input"]
        bert_out = sample["bert_output"]

        # print(sim, "hi there", len(bert_out))
        # sim_out = bert_out[random.choice(sim)]
        sim_out = global_bert_output[random.choice(sim)]

        # Process input
        in_len.append(min(len(input_s), max_len))
        src[i, :in_len[-1]] = torch.tensor(input_s[:in_len[-1]])

        # Process output
        out_len.append(min(len(output_s), max_len))
        content_trg[i, :out_len[-1]] = torch.tensor(output_s[:out_len[-1]])
        content_len.append(out_len[-1])

        trg[i, :out_len[-1]] = torch.tensor(output_s[:out_len[-1]])
        trg[i, out_len[-1]] = 3  # EOS token
        trg_input[i, 1:out_len[-1] + 1] = torch.tensor(output_s[:out_len[-1]])
        trg_input[i, 0] = 2  # SOS token

        # Process BERT inputs/outputs
        bert_src[i, :min(len(bert_in), max_len + 2)] = torch.tensor(bert_in[:max_len + 2])
        bert_trg[i, :min(len(bert_out), max_len + 2)] = torch.tensor(bert_out[:max_len + 2])
        bert_sim[i, :min(len(sim_out), max_len + 2)] = torch.tensor(sim_out[:max_len + 2])

    return (
        src,
        torch.tensor(in_len),
        trg,
        trg_input,
        torch.tensor(out_len),
        bert_src,
        bert_trg,
        bert_sim,
        content_trg,
        torch.tensor(content_len)
    )

if __name__ == "__main__":
    train_data = STdata("train")
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for i, batch in enumerate(train_loader):
        print(batch)
        break