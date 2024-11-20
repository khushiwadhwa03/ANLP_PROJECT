import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, GPT2Model, RobertaModel
from tqdm import tqdm

class StyleExtractor(nn.Module):
    def __init__(self,config_dict):
        super().__init__()
        self.model = BertModel.from_pretrained("bert-base-uncased",output_hidden_states=True,cache_dir='.')

    def forward(self,input):
        outputs = self.model(input)
        hidden_states = torch.stack(outputs[2],dim=1)
        first_hidden_states = hidden_states[:,:,0,:]
        #print(first_hidden_states.shape)[64,13,768]
        #print(outputs[0][:,0,:])
        #print(first_hidden_states[:,-1,:])
        return first_hidden_states

# class GPTStyleExtractor(nn.Module):
#     def __init__(self, config_dict):
#         super().__init__()
#         self.model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True, cache_dir='.')

#     def forward(self, input):
#         outputs = self.model(input)
#         hidden_states = torch.stack(outputs.hidden_states, dim=1)  # GPT-2 has `hidden_states` instead of the tuple
#         first_hidden_states = hidden_states[:, :, 0, :]  # [batch_size, seq_len, 768]
#         return first_hidden_states

# class RoBERTAStyleExtractor(nn.Module):
#     def __init__(self, config_dict):
#         super().__init__()
#         self.model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True, cache_dir='.')

#     def forward(self, input):
#         outputs = self.model(input)
#         hidden_states = torch.stack(outputs.hidden_states, dim=1)  # Extract all hidden states
#         first_hidden_states = hidden_states[:, :, 0, :]  # Select the first token's hidden states (e.g., <s>)
        
#         # Print shapes for debugging if needed
#         # print(first_hidden_states.shape)  # Expected shape: [batch_size, num_layers, hidden_size]
        
#         return first_hidden_states
