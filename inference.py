import json
import pickle
import torch
import os
import argparse
import nltk

from model import Seq2Seq
from style import StyleExtractor
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--dataset', type=str, default='para', choices=['para', 'quora'])
    parser.add_argument('--max_len', type=int, default=15)
    parser.add_argument('--model_save_path', type=str, default="/ssd_scratch/cvit/akshat/quora_random")
    parser.add_argument('--idx', type=int, default=30)
    parser.add_argument('--idx2', type=int, default=4)

    opt = parser.parse_args()

    if opt.dataset == 'quora':
        opt.data_folder = './data/'
        opt.config="config.json"
    elif opt.dataset == 'para':
        opt.data_folder = './data2/'
        opt.config="config2.json"
    return opt


def getword(idx2word,lists):
    results = []
    for list in lists:
        append = False
        result = []
        for idx in list:
            if idx == 0 or idx == 2:
                continue
            if idx == 3:
                append = True
                results.append(result)
                break
            result.append(idx2word[idx.item()])
        if not append:
            results.append(result)
    return results

opt = parse_option()
with open(opt.config) as f:
    config = json.load(f)

seq2seq = Seq2Seq(config).to(device)
stex = StyleExtractor(config).to(device)


seq2seq.load_state_dict(torch.load(os.path.join(opt.model_save_path,"seq2seq"+str(opt.idx)+".pkl")))
stex.load_state_dict(torch.load(os.path.join(opt.model_save_path,"stex"+str(opt.idx)+".pkl")))

seq2seq.eval()
stex.eval()
seq2seq.decoder.mode = "infer"

with open(os.path.join(opt.data_folder,'idx2word.pkl'), 'rb') as f:
    idx2word = pickle.load(f)
with open(os.path.join(opt.data_folder,'word2idx.pkl'), 'rb') as f:
    word2idx = pickle.load(f)
with open(os.path.join(opt.data_folder,'test/bert_src.pkl'),'rb') as f:
    bert_output = pickle.load(f)
with open(os.path.join(opt.data_folder,'test/trg.pkl'),'rb') as f:
    normal_output = pickle.load(f)


# src = ["how do i develop good project management skills ?"]
# gen - how can a project management ideas done for skills ?
# exm - why do some people like cats more than dogs ?
# src = ["which is the best anime to watch ?"]
# src = ["if you are at the lowest point of your life , what do you do ?"]
#src = ["when did you first realize that you were gay lesbian bi ?"]

src = ["he believed his son had died in a terrorist attack ."]
# src = ["it is hard for me to imagine where they could be hiding it underground ."]
#src = ["do you want to kiss teddy ?"]
# src = ["i had a strange call from a woman ."]
def pack_sim(sim):
    bert_sim = torch.zeros(opt.max_len + 2, dtype=torch.long)
    bert_sim[0:min(len(sim), opt.max_len + 2)] = torch.tensor(sim[0:min(opt.max_len + 2, len(sim))])
    return bert_sim
def pack_input(src):
    src_tokennized = [nltk.word_tokenize(line) for line in src]
    src_idx = [[word2idx[word] if word in word2idx else word2idx['UNK'] for word in words] for words in src_tokennized]
    src_idx = src_idx[0]
    in_len = len(src_idx)
    src_out = torch.zeros(opt.max_len, dtype=torch.long)

    if in_len > opt.max_len:
        src_out[0:opt.max_len] = torch.tensor(src_idx[0:opt.max_len])
        in_len = opt.max_len
    else:
        src_out[0:in_len] = torch.tensor(src_idx)

    src_out = src_out.unsqueeze(0)
    length = torch.tensor([in_len])
    return src_out, length

src_idx, in_len = pack_input(src)


with torch.no_grad():
    count = 0
    temp_arr = []
    src_idx = src_idx.to(device)
    in_len = in_len.to(device)
    cnt = 0
    for style in bert_output:
        bert_sim = pack_sim(style)
        bert_sim = bert_sim.unsqueeze(0)
        bert_sim = bert_sim.to(device)
        style_emb = stex(bert_sim)
        id_arr,_ = seq2seq.forward(src_idx, in_len, style_emb)
        temp_arr.append(id_arr)
        cnt+=1
        # if (cnt == 5):
        #     break
       


    filename = os.path.join(opt.model_save_path,"gen"+str(opt.idx2)+".txt")
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'a') as f:
        for bat in temp_arr:
            ss = getword(idx2word, bat)
            for s in ss:
                f.write(' '.join(s))
                f.write('\n')

