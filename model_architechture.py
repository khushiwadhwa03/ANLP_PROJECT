import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
# from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # the encoder is just a GRU, dont forget to add bidirectional
        self.rnn = nn.GRU(
            input_size = 300,
            hidden_size = 512, # hdim of the encoder
            num_layers = 1, # layers, encoder
            dropout = 0,
            batch_first=True, # important TODO check later
            bidirectional=True # bidirectional TODO important
        )

    def forward(self, input_seq, seq_lengths):
        sorted_lengths, sort_indices = seq_lengths.sort(dim=-1, descending=True)
        sorted_seq = input_seq.index_select(0, sort_indices)

        # padding
        packed_input_gru = pack_padded_sequence(sorted_seq, sorted_lengths, batch_first=True) # again using batch first convention, important # TODO

        packed_output, hidden_states = self.rnn(packed_input_gru)
        output_seq, _ = pad_packed_sequence(packed_output, batch_first=True)

        hidden_states = torch.cat(hidden_states, dim=-1)
        inverse_indices = sort_indices.argsort(dim=-1, descending=False)
        
        output_seq = output_seq.index_select(0, inverse_indices)
        hidden_states = hidden_states.index_select(0, inverse_indices)
        return output_seq, hidden_states


class ScaleDotAttention(nn.Module):
    def __init__(self, mode='Dot'):
        super().__init__()
        decoder_dim = 512# corresponding to the query
        encoder_dim = 1024 # key dim

        if mode == 'Self':
            decoder_dim = encoder_dim

        self.attn_weights = nn.Parameter(torch.randn([decoder_dim, encoder_dim], device=device))
        nn.init.normal_(self.attn_weights, mean=0., std=np.sqrt(2. / (decoder_dim + encoder_dim)))

    def forward(self, query, key, value, mask):
        # using the masked attention formulation
        attention_scores = key.bmm(query.mm(self.attn_weights).unsqueeze(dim=2)).squeeze(dim=2)
        
        attention_scores.masked_fill_(mask[:, :attention_scores.size(-1)], -float('inf')) # masked attention
        attention_scores = attention_scores.softmax(dim=-1)
        
        attention_output = (attention_scores.unsqueeze(dim=2) * value).sum(dim=1)

        return attention_output, attention_scores


class Attention(nn.Module):
    def __init__(self, key_dim, value_dim, output_dim):
        super().__init__()
        
        # defining the linear layers, can be used later itself
        self.attention_linear = nn.Linear(key_dim, value_dim)
        self.output_projection = nn.Linear(value_dim, output_dim)

    def forward(self, keys, values):
        # using the attention mechanism to get the context vector
        keys = torch.transpose(self.attention_linear(torch.transpose(keys, 0, 1)), 0, 1)
        attention_weights = torch.matmul(keys, values).transpose(1, 2)
        attended_values = torch.matmul(values, attention_weights).squeeze(2) # squeeze the last dimension
        return self.output_projection(attended_values)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_projection = nn.Linear(1792, 512) # 1792 si coming from the encoder and style embedding, 1024 from encoder and 768 from style embedding
        self.word_embedding = None
        self.attention = ScaleDotAttention()
        self.rnn = nn.GRU(
            input_size=1324, # 512 from the hidden state, 512 from the context vector, 300 from the word embedding (?)
            hidden_size=512,
            batch_first=True
        )
        self.output_projection = nn.Linear(512, 25_000) # vocab size for now, randomize it later
        self.style_attention = Attention(512, 768, 300)
        self.mode = 'train'

    def forward(self, encoder_hidden, encoder_outputs, encoder_mask, decoder_input, style_embedding):
        
        # rn this code is only train and eval, not for doing inference # TODO important

        style_vector = style_embedding[:, -1, :]
        combined_hidden = torch.cat([encoder_hidden, style_vector], dim=-1)
        hidden_state = self.hidden_projection(combined_hidden).unsqueeze(0)

        decoder_input_emb = self.word_embedding(decoder_input)
        outputs = []

        for idx in range(decoder_input.size(1)):
            # we later need to define a stopping criterion stop at some maximum length # TODO important
            context_vector, _ = self.attention(hidden_state[-1], encoder_outputs, encoder_outputs, encoder_mask)
            output, hidden_state = self.rnn(torch.cat([context_vector, decoder_input_emb[:, idx]], dim=-1).unsqueeze(dim=1), hidden_state)
            
            outputs.append(output.squeeze(dim=1))

        final_output = self.output_projection(torch.stack(outputs, dim=1))
        return final_output


class Seq2Seq(nn.Module):
    def __init__(self):

        super().__init__()

        embedding_dim = 300
        self.emb_layer = nn.Embedding(num_embeddings = 25_000, embedding_dim = embedding_dim, padding_idx=0)
        # print("glove embeddings start")
        # we need to get the glove embeddings, but rn we can use random embeddings
        word_embeddings = np.random.rand(25_000, embedding_dim) # TODO important
        # print("glove embeddings end")
        self.emb_layer.weight.data.copy_(torch.from_numpy(word_embeddings)) # initalize the embedding layer
        self.para_req_grad = filter(lambda x: x.requires_grad, self.parameters()) # filtering out the parameters that require grad
        print('Word Vectors initialized')

        # initializing the components
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.decoder.word_embedding = self.emb_layer # already has the data from glove

        # hyperparmeter to be tuned # IMPORTANT TODO adam, sgd, adamw testing, and also lr
        self.opt = Adam(params = list(self.para_req_grad), lr = 1e-4)

    def forward(self, sequence, seq_len, style_emb, decoder_input=None):

        # after initial embedding is encoded, we now decode that output to get the final output for the entire model (we also have the style embedding)
        sequence = self.emb_layer(sequence)
        encoded_feature, encode_hidden = self.encoder(sequence, seq_len)
        out = self.decoder(encode_hidden, encoded_feature, (sequence == 0).byte(), decoder_input, style_emb)
        return out


if __name__ == '__main__':

    # later use config.json, need to have a different one for different dataloader and diff model configuration styles
    model = Seq2Seq()
    model.to(device)
    print(model)