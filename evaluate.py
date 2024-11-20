import json
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from dataset import STdata, collate_fn
from model import Seq2Seq
from style import StyleExtractor
from transformers import BertTokenizer
import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--dataset', type=str, default='quora', choices=['para', 'quora'])
    parser.add_argument('--max_len', type=int, default=15)
    parser.add_argument('--model_save_path', type=str, default="/ssd_scratch/cvit/akshat/quora_random")
    parser.add_argument('--idx', type=int, default=45)

    opt = parser.parse_args()

    if opt.dataset == 'quora':
        opt.data_folder = './data/'
        opt.config="config_quora.json"

    elif opt.dataset == 'para':
        opt.data_folder = './data2/'
        opt.config="config2_para.json"

    return opt

opt = parse_option()

with open(opt.config) as f:
    print("Using the config, ",opt.config)
    config = json.load(f)


test_set = STdata("test", dataroot=opt.data_folder, max_len=15)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=lambda x: collate_fn(x, max_len=opt.max_len))


seq2seq = Seq2Seq(config, opt).to(device)
stex = StyleExtractor(config).to(device)

seq2seq.load_state_dict(torch.load(os.path.join(opt.model_save_path,"seq2seq"+str(opt.idx)+".pkl")))
stex.load_state_dict(torch.load(os.path.join(opt.model_save_path,"stex"+str(opt.idx)+".pkl")))

seq2seq.eval()
stex.eval()
seq2seq.decoder.mode = "infer"

with open(os.path.join(opt.data_folder,'test/sim.pkl'),'rb') as f:
    sim = pickle.load(f)
with open(os.path.join(opt.data_folder,'idx2word.pkl'), 'rb') as f:
    idx2word = pickle.load(f)
with open(os.path.join(opt.data_folder,'test/bert_trg.pkl'),'rb') as f:
    bert_output = pickle.load(f)

with open(os.path.join(opt.data_folder,'test/trg.pkl'),'rb') as f:
    normal_output = pickle.load(f)

with torch.no_grad():
    arr = []
    examplar_list =[]
    count = 0
    for src,in_len,trg,trg_input,ou_len,bert_src,bert_trg,bert_sim,content_trg,content_len in tqdm(test_loader):
        src, in_len, trg, trg_input, ou_len, bert_src, bert_trg, bert_sim,content_trg,content_len = \
            src.to(device), in_len.to(device), trg.to(device), trg_input.to(device), ou_len.to(device) \
                , bert_src.to(device), bert_trg.to(device), bert_sim.to(
                device),content_trg.to(device),content_len.to(device)
        temp_arr = []
        for s_idx in sim[count]:
            bert_sim = torch.zeros(15 + 2, dtype=torch.long)
            bert_sim_ = bert_output[s_idx]
            bert_sim[0:min(len(bert_sim_), 15 + 2)] = torch.tensor(bert_sim_[0:min(15 + 2, len(bert_sim_))])
            bert_sim = bert_sim.unsqueeze(0)
            bert_sim = bert_sim.to(device)
            style_emb = stex(bert_sim)
            
            id_arr,_ = seq2seq.forward(src, in_len, style_emb, response=trg, decoder_input=trg_input)
            temp_arr.append(id_arr)
       
        right = -1

        max_cover = -1

        for aa,aaa in enumerate(temp_arr):

            if len(set(normal_output[count]) & set([ii.item() for ii in aaa[0]]))/len(set(normal_output[count]))>max_cover:
                max_cover = len(set(normal_output[count]) & set([ii.item() for ii in aaa[0]]))/len(set(normal_output[count]))
                right = aa

        examplar_list.append(sim[count][right])
        arr.append(temp_arr[right])
        count+=1

    filename = os.path.join(opt.model_save_path,"trg_gen"+str(opt.idx)+".txt")
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'a') as f:
        for bat in arr:
            ss = getword(idx2word, bat)
            for s in ss:
                f.write(' '.join(s))
                f.write('\n')
    exm_file = os.path.join(opt.model_save_path,"exm"+str(opt.idx)+".txt")
    with open(exm_file,'a') as f:
        for ii in examplar_list:
            f.write(tokenizer.decode(bert_output[ii][1:-2]))
            f.write('\n')
