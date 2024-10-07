import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import json
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        encoder_config = config_dict['encoder']
        self.rnn = nn.GRU(
            input_size=encoder_config.get('input_dim', 256), # by defalt 256 if nothing is provided
            hidden_size=encoder_config.get('hidden_dim', 256), # hdim of the encoder
            num_layers=encoder_config.get('num_layers', 1),
            dropout=0 if encoder_config.get('num_layers', 1) == 1 else config_dict.get('drop_out', 0.2),
            batch_first=True, # important TODO check later
            bidirectional=encoder_config.get('bidirectional', True)
        )

    def forward(self, input_seq, seq_lengths):
        sorted_lengths, sort_indices = seq_lengths.sort(dim=-1, descending=True)
        sorted_seq = input_seq.index_select(0, sort_indices)
        packed_input_gru = pack_padded_sequence(sorted_seq, sorted_lengths, batch_first=True) # again using batch first convention, important # TODO

        packed_output, hidden_states = self.rnn(packed_input_gru)

        output_seq, _ = pad_packed_sequence(packed_output, batch_first=True)
        hidden_states = torch.cat(hidden_states, dim=-1)
        inverse_indices = sort_indices.argsort(dim=-1, descending=False)
        output_seq = output_seq.index_select(0, inverse_indices)
        hidden_states = hidden_states.index_select(0, inverse_indices)
        return output_seq, hidden_states


class ScaleDotAttention(nn.Module):
    def __init__(self, config_dict, mode='Dot'):
        super().__init__()
        decoder_dim = config_dict['decoder'].get('hidden_dim', 256)# corresponding to the query
        encoder_dim = config_dict['encoder'].get('final_out_dim', 256) # key dim
        if mode == 'Self':
            decoder_dim = encoder_dim
        self.attn_weights = nn.Parameter(torch.randn([decoder_dim, encoder_dim], device=device))
        nn.init.normal_(self.attn_weights, mean=0., std=np.sqrt(2. / (decoder_dim + encoder_dim)))

    def forward(self, query, key, value, mask, bias=None):
        attention_scores = key.bmm(query.mm(self.attn_weights).unsqueeze(dim=2)).squeeze(dim=2)
        if bias is not None:
            attention_scores += bias
        attention_scores.masked_fill_(mask[:, :attention_scores.size(-1)], -float('inf'))
        attention_scores = attention_scores.softmax(dim=-1)
        attention_output = (attention_scores.unsqueeze(dim=2) * value).sum(dim=1)
        return attention_output, attention_scores


class Attention(nn.Module):
    def __init__(self, key_dim, value_dim, output_dim):
        super().__init__()
        self.attention_linear = nn.Linear(key_dim, value_dim)
        self.output_projection = nn.Linear(value_dim, output_dim)

    def forward(self, keys, values):
        keys = torch.transpose(self.attention_linear(torch.transpose(keys, 0, 1)), 0, 1)
        attention_weights = torch.matmul(keys, values).transpose(1, 2)
        attended_values = torch.matmul(values, attention_weights).squeeze(2)
        return self.output_projection(attended_values)


class DecoderCell(nn.Module):
    def __init__(self, config_dict, is_auxiliary=False):
        super().__init__()
        decoder_config = config_dict['decoder']
        input_size = decoder_config.get('input_dim', 256)
        if is_auxiliary:
            input_size += decoder_config.get('hidden_dim', 256)
        self.rnn_cell = nn.GRU(input_size=input_size, hidden_size=decoder_config.get('hidden_dim', 256), batch_first=True)

    def forward(self, decoder_input, prev_hidden_state):
        output, hidden_state = self.rnn_cell(decoder_input.unsqueeze(1), prev_hidden_state)
        return output.squeeze(1), hidden_state


class Decoder(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        encoder_config = config_dict['encoder']
        decoder_config = config_dict['decoder']
        self.hidden_projection = nn.Linear(
            encoder_config.get('final_out_dim', 256) + config_dict['style_attn']['style_in'], 
            decoder_config.get('hidden_dim', 256)
        )
        self.word_embedding = None
        self.attention = ScaleDotAttention(config_dict)
        self.rnn = nn.GRU(
            input_size=decoder_config.get('input_dim', 256),
            hidden_size=decoder_config.get('hidden_dim', 256),
            batch_first=True
        )
        self.output_projection = nn.Linear(decoder_config.get('hidden_dim', 256), config_dict.get('vocabulary_dim', 1))
        self.style_attention = Attention(decoder_config.get('hidden_dim', 256), 
                                         config_dict['style_attn']['style_in'], 
                                         config_dict['style_attn']['style_out'])
        self.mode = 'train'

    def forward(self, encoder_hidden, encoder_outputs, encoder_mask, target_seq, decoder_input, style_embedding, max_seq_len=21):
        style_vector = style_embedding[:, -1, :]
        combined_hidden = torch.cat([encoder_hidden, style_vector], dim=-1)
        hidden_state = self.hidden_projection(combined_hidden).unsqueeze(0)

        if self.mode in ['train', 'eval']:
            decoder_input_emb = self.word_embedding(decoder_input)
            outputs = []

            for t in range(decoder_input.size(1)):
                context_vector, _ = self.attention(hidden_state[-1], encoder_outputs, encoder_outputs, encoder_mask)
                rnn_input = torch.cat([context_vector, decoder_input_emb[:, t]], dim=-1).unsqueeze(dim=1)
                output, hidden_state = self.rnn(rnn_input, hidden_state)
                outputs.append(output.squeeze(dim=1))

            final_output = self.output_projection(torch.stack(outputs, dim=1))
            return final_output

        elif self.mode == 'infer':
            generated_ids = []
            prev_token_emb = self.word_embedding(
                torch.full((encoder_outputs.size(0),), 2, dtype=torch.long, device=device)
            )

            for _ in range(max_seq_len):
                context_vector, _ = self.attention(hidden_state[-1], encoder_outputs, encoder_outputs, encoder_mask)
                rnn_input = torch.cat([context_vector, prev_token_emb], dim=-1).unsqueeze(dim=1)
                output, hidden_state = self.rnn(rnn_input, hidden_state)
                logits = self.output_projection(output.squeeze(dim=1))
                predicted_token_id = logits.argmax(dim=-1)
                prev_token_emb = self.word_embedding(predicted_token_id)
                generated_ids.append(predicted_token_id)

            return torch.stack(generated_ids, dim=1)

        return None


class Seq2Seq(nn.Module):
    def __init__(self, config_dict):

        super().__init__()

        conf_embedding_dim = config_dict.get('embedding_dim', 1) # by default 1, but 300 in config
        self.emb_layer = nn.Embedding(num_embeddings = config_dict.get('vocabulary_dim', 1), embedding_dim = conf_embedding_dim, padding_idx=0)
        # print("glove embeddings start")
        # we need to get the glove embeddings, but rn we can use random embeddings
        word_embeddings = np.random.rand(config_dict.get('vocabulary_dim', 1), conf_embedding_dim) # TODO important
        # print("glove embeddings end")
        self.emb_layer.weight.data.copy_(torch.from_numpy(word_embeddings)) # initalize the embedding layer
        self.para_req_grad = filter(lambda x: x.requires_grad, self.parameters()) # filtering out the parameters that require grad
        print('Word Vectors initialized')

        # initializing the components
        self.encoder = Encoder(config_dict=config_dict)
        self.decoder = Decoder(config_dict)
        self.decoder.word_emb_layer = self.emb_layer # already has the data from glove

        # hyperparmeter to be tuned # IMPORTANT TODO adam, sgd, adamw testing, and also lr
        self.opt = Adam(params=list(self.para_req_grad), lr=config_dict.get('lr', 1e-4))

    def forward(self, sequence, seq_len, style_emb, decoder_input=None, max_seq_len=16):

        # after initial embedding is encoded, we now decode that output to get the final output for the entire model (we also have the style embedding)
        encode_mask = (sequence == 0).byte() # or bool
        sequence = self.emb_layer(sequence)
        encode_output, encode_hidden = self.encoder(sequence, seq_len)
        all_output = self.decoder(encode_hidden, encode_output, encode_mask, decoder_input,style_emb,max_seq_len=max_seq_len)
        return all_output, encode_hidden


if __name__ == '__main__':

    # using config.json for now, need to have a different one for different dataloader
    with open("configs/config.json") as f:
        config_dict = json.load(f)

    model = Seq2Seq(config_dict)
    model.to(device)
    print(model)