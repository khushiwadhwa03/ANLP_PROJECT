import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import numpy as np
import pickle
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialise_word_embedding(config_dict):
    file_path = "./data2/word2idx.pkl" # by default for para
    if (config_dict.get('dataset') == 'quora'):
        file_path = "./data/word2idx.pkl"

    with open(file_path, 'rb') as f:
        vocab = pickle.load(f)
    
    print("loading glove model")
    
    f = open("glove.6B.300d.txt", 'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding

    print("total number of words loaded from glove: ",len(gloveModel))
        
    # dimension of the glove embedding - 300
    word_emb = np.zeros((len(vocab), 300))
    missing = 0
    for word,idx in vocab.items():
        if word in gloveModel:
            word_emb[idx] = gloveModel[word]
            continue
        missing += 1

    print(str(missing), ": number of words not in glove embedding")

    return word_emb


def get_nll_loss(fc_out, fc_label, reduction='mean', ope='log_softmax'):
    eps = 1e-6
    fc_out = F.log_softmax(fc_out, dim=-1)
    label = (fc_label > 0).float()
    
    if len(fc_label.size()) == 1:
        return label * F.nll_loss(input=fc_out, target=fc_label, reduction=reduction)
    
    elif len(fc_label.size()) == 2:
        loss = (label * F.nll_loss(input=fc_out.transpose(1, 2), target=fc_label, reduction='none')).sum(dim=-1) / (label.sum(dim=-1) + eps)
        if reduction == 'mean':
            return loss.mean()
        else:
            return loss



class Encoder(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        hidden_dim = config_dict['encoder'].get('hidden_dim', 256) # 512
        input_dim = config_dict['encoder'].get('input_dim', 256) # 300
        bidirectional = bool(config_dict['encoder'].get('bidirectional', True))
        # bidirectional = False
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, bias=True, batch_first=True, bidirectional=bidirectional)


    def forward(self, seq_arr, seq_len):

        seq_len_sorted, index = seq_len.sort(dim=-1, descending=True)
        seq_arr_sorted = seq_arr.index_select(0, index)
        padded_input = pack_padded_sequence(seq_arr_sorted, seq_len_sorted.cpu(), batch_first=True) # padded the input which is sorted
        output, hidden = self.rnn(padded_input)
        output, _ = pad_packed_sequence(output, batch_first=True)
        hidden = torch.cat(tuple(hidden), dim=-1)
        _, inverse_index = index.sort(dim=-1, descending=False)
        output = output.index_select(0, inverse_index)
        hidden = hidden.index_select(0, inverse_index)
        # print(output.shape, hidden.shape, "shapes for important things")
        # torch.Size([128, 15, 1024]) torch.Size([128, 1024])
        return output, hidden


class TransformerCLSEncoder(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        hidden_dim = config_dict['encoder'].get('hidden_dim', 256)
        num_layers = 1
        nhead = config_dict['encoder'].get('nhead', 4)
        dropout = config_dict.get('drop_out', 0.2)
        input_dim = config_dict['encoder'].get('input_dim', 256)
        output_dim = 1024

        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.output_projection = nn.Linear(input_dim, output_dim)

    def forward(self, seq_arr, seq_len):
        batch_size, _, _ = seq_arr.size()
        
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        seq_arr = torch.cat([cls_token, seq_arr], dim=1)
        
        seq_arr = seq_arr + self.positional_encoding[:, :seq_arr.size(1), :]
        transformer_output = self.transformer_encoder(seq_arr)
        
        projected_output = self.output_projection(transformer_output)
        
        cls_output = projected_output[:, 0, :]
        output = projected_output[:, 1:, :]
        
        # print("shapes of transformer outputs", projected_output.shape, cls_output.shape)
        # torch.Size([128, 16, 1024]) torch.Size([128, 1024])
        return output, cls_output

class TransformerAVGEncoder(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        hidden_dim = config_dict['encoder'].get('hidden_dim', 256)
        num_layers = 1
        nhead = config_dict['encoder'].get('nhead', 4)
        dropout = config_dict.get('drop_out', 0.2)
        input_dim = config_dict['encoder'].get('input_dim', 256)
        output_dim = 1024
        
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(input_dim, output_dim)

    def forward(self, seq_arr, seq_len):
        seq_arr = seq_arr + self.positional_encoding[:, :seq_arr.size(1), :]
        transformer_output = self.transformer_encoder(seq_arr)
        projected_output = self.output_projection(transformer_output)
        # averaging the sequence output along the time dimension
        averaged_output = projected_output.mean(dim=1)
        
        return projected_output, averaged_output


# Encoder LSTM
class LSTMEncoder(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        hidden_dim = config_dict['encoder'].get('hidden_dim', 256)
        input_dim = config_dict['encoder'].get('input_dim', 256)
        bidirectional = bool(config_dict['encoder'].get('bidirectional', True))
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, bias=True,
                           batch_first=True, bidirectional=bidirectional)

    def forward(self, seq_arr, seq_len):
        seq_len_sorted, index = seq_len.sort(dim=-1, descending=True)
        seq_arr_sorted = seq_arr.index_select(0, index)
        padded_input = pack_padded_sequence(seq_arr_sorted, seq_len_sorted.cpu(), batch_first=True)
        output, (hidden, _) = self.rnn(padded_input)
        output, _ = pad_packed_sequence(output, batch_first=True)

        hidden = torch.cat(tuple(hidden), dim=-1)
        _, inverse_index = index.sort(dim=-1, descending=False)
        output = output.index_select(0, inverse_index)
        hidden = hidden.index_select(0, inverse_index)

        return output, hidden


class ScaleDotAttention(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        
        dim_q = config_dict['decoder'].get('hidden_dim', 256)
        dim_k = config_dict['encoder'].get('final_out_dim', 256)
        
        self.W = nn.Parameter(torch.empty([dim_q, dim_k], device=device, requires_grad=True), requires_grad=True)
        nn.init.normal_(self.W, mean=0., std=np.sqrt(2. / (dim_q + dim_k)))

    def forward(self, q, k, v, mask, bias=None):
        attn_weight = k.bmm(q.mm(self.W).unsqueeze(dim=2)).squeeze(dim=2)
        if bias is not None:
            attn_weight += bias

        mask = mask[:,:attn_weight.shape[-1]].bool()
        attn_weight.masked_fill(mask, - float('inf'))
        attn_weight = attn_weight.softmax(dim=-1)
        attn_out = (attn_weight.unsqueeze(dim=2) * v).sum(dim=1)
        return attn_out, attn_weight



class Decoder(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        encoder_final_out_dim = config_dict['encoder'].get('final_out_dim', 256) # 1024
        decoder_hidden_dim = config_dict['encoder'].get('hidden_dim', 256)
        vocabulary_dim = config_dict.get('vocabulary_dim', 1)
        input_dim = config_dict['decoder'].get('input_dim', 256)
        hidden_dim = config_dict['decoder'].get('hidden_dim', 256)

        self.W_e2d = nn.Linear(encoder_final_out_dim + config_dict['style_in'], decoder_hidden_dim, bias=True) # 1024+768
        self.word_emb_layer = None
        self.attention_layer = ScaleDotAttention(config_dict)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bias=True, batch_first=True, bidirectional=False)
        self.projection_layer = nn.Linear(hidden_dim, vocabulary_dim)
        self.mode = 'train'

    def forward(self, encode_hidden, encode_output, encoder_mask, seq_label, decoder_input, style_emb, max_seq_len=21):
        style_feature = style_emb[:, -1, :]
        encode_hidden = torch.cat([encode_hidden,style_feature],dim=-1)

        # print("Encode hidden shape", encode_hidden.shape)
        # (128, 1068) when using transformer encoder # 1068 - 768
        # (128, 1792) when using GRU encoder # 1792 - 256*7
        hidden = self.W_e2d(encode_hidden).unsqueeze(0)

        if self.mode in ['train', 'eval']:
            decoder_input_emb = self.word_emb_layer(decoder_input)
            decoder_output_arr = []

            for t in range(decoder_input.size()[-1]):
                # we later need to define a stopping criterion stop at some maximum length # TODO important
                context, _ = self.attention_layer(hidden[-1], encode_output, encode_output, encoder_mask)
                output, hidden = self.gru(torch.cat([context,decoder_input_emb[:, t]], dim=-1).unsqueeze(dim=1), hidden)
                decoder_output_arr.append(output.squeeze(dim=1))
            decoder_output = self.projection_layer(torch.stack(decoder_output_arr, dim=1))
            if self.mode == 'eval':
                loss = get_nll_loss(decoder_output, seq_label, reduction='none')
                ppl = loss.exp()
                return ppl, loss.mean()
            else:
                loss = get_nll_loss(decoder_output, seq_label, reduction='mean')
                return loss
        elif self.mode == 'infer':
            id_arr = []
            previous_vec = self.word_emb_layer(
                torch.ones(size=[encode_output.size()[0]], dtype=torch.long, device=device) *
                torch.tensor(2,  dtype=torch.long, device=device))
            for t in range(max_seq_len):
                context, _ = self.attention_layer(hidden[-1], encode_output, encode_output, encoder_mask)
                output, hidden = self.gru(torch.cat([context,previous_vec], dim=-1).unsqueeze(dim=1), hidden)
                decode_output = self.projection_layer(output.squeeze(dim=1))
                _, previous_id = decode_output.max(dim=-1, keepdim=False)
                previous_vec = self.word_emb_layer(previous_id)
                id_arr.append(previous_id)
            decoder_id = torch.stack(id_arr, dim=1)
            return decoder_id
        else:
            return None


# Decoder LSTM
class LSTMDecoder(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        encoder_final_out_dim = config_dict['encoder'].get('final_out_dim', 256)
        decoder_hidden_dim = config_dict['encoder'].get('hidden_dim', 256)
        vocabulary_dim = config_dict.get('vocabulary_dim', 1)
        input_dim = config_dict['decoder'].get('input_dim', 256)
        hidden_dim = config_dict['decoder'].get('hidden_dim', 256)
        self.W_e2d = nn.Linear(encoder_final_out_dim + config_dict['style_in'], decoder_hidden_dim, bias=True)
        self.word_emb_layer = None
        self.attention_layer = ScaleDotAttention(config_dict)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bias=True, batch_first=True, bidirectional=False)
        self.projection_layer = nn.Linear(hidden_dim, vocabulary_dim)
        self.mode = 'train'

    def forward(self, encode_hidden, encode_output, encoder_mask, seq_label, decoder_input, style_emb, max_seq_len=21):
        style_feature = style_emb[:, -1, :]
        encode_hidden = torch.cat([encode_hidden, style_feature], dim=-1)

        hidden = self.W_e2d(encode_hidden).unsqueeze(0)
        cell = torch.zeros_like(hidden)

        if self.mode in ['train', 'eval']:
            decoder_input_emb = self.word_emb_layer(decoder_input)
            decoder_output_arr = []

            for t in range(decoder_input.size()[-1]):
                context, _ = self.attention_layer(hidden[-1], encode_output, encode_output, encoder_mask)
                output, (hidden, cell) = self.lstm(torch.cat([context, decoder_input_emb[:, t]], dim=-1).unsqueeze(dim=1), (hidden, cell))
                decoder_output_arr.append(output.squeeze(dim=1))
            decoder_output = self.projection_layer(torch.stack(decoder_output_arr, dim=1))
            if self.mode == 'eval':
                loss = get_nll_loss(decoder_output, seq_label, reduction='none')
                ppl = loss.exp()
                return ppl, loss.mean()
            else:
                loss = get_nll_loss(decoder_output, seq_label, reduction='mean')
                return loss
        elif self.mode == 'infer':
            id_arr = []
            previous_vec = self.word_emb_layer(
                torch.ones(size=[encode_output.size()[0]], dtype=torch.long, device=device) *
                torch.tensor(2, dtype=torch.long, device=device))
            for t in range(max_seq_len):
                context, _ = self.attention_layer(hidden[-1], encode_output, encode_output, encoder_mask)
                output, (hidden, cell) = self.lstm(torch.cat([context, previous_vec], dim=-1).unsqueeze(dim=1), (hidden, cell))
                decode_output = self.projection_layer(output.squeeze(dim=1))
                _, previous_id = decode_output.max(dim=-1, keepdim=False)
                previous_vec = self.word_emb_layer(previous_id)
                id_arr.append(previous_id)
            decoder_id = torch.stack(id_arr, dim=1)
            return decoder_id
        else:
            return None




class Seq2Seq(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        embedding_dim = config_dict.get('embedding_dim', 1)
        vocabulary_dim = config_dict.get('vocabulary_dim', 1)
        self.emb_layer = nn.Embedding(num_embeddings=vocabulary_dim, embedding_dim=embedding_dim, padding_idx=0)
        print('glove - initializing word vectors')
        glove_weight = initialise_word_embedding(config_dict)
        print(glove_weight.shape, self.emb_layer.weight.data.shape)
        self.emb_layer.weight.data.copy_(torch.from_numpy(glove_weight))

        print('initialized glove')

        self.encoder_layer = Encoder(config_dict)
        self.decoder = Decoder(config_dict)
        self.decoder.word_emb_layer = self.emb_layer
        self.para = list(filter(lambda x: x.requires_grad, self.parameters()))
        # hyperparmeter to be tuned # IMPORTANT TODO adam, sgd, adamw testing, and also lr
        self.opt = Adam(params=self.para, lr=config_dict.get('lr', 1e-4))

    def forward(self, seq_arr, seq_len,style_emb, response=None, decoder_input=None, max_seq_len=16):

        encode_mask = (seq_arr == 0).byte()
        seq_arr = self.emb_layer(seq_arr) # x ki glove embedding
        encode_output, encode_hidden = self.encoder_layer(seq_arr, seq_len)
        all_output = self.decoder(encode_hidden, encode_output, encode_mask, response,
                                  decoder_input,style_emb,max_seq_len=max_seq_len)
        return all_output, encode_hidden





   