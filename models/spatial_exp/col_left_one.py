from .attention import MultiHeadAttention,ScaledDotProductAttention
from .encoders import EncoderLayer
from .decoders import DecoderLayer
from .transformer import Transformer
from models.transformer.utils import sinusoid_encoding_table
from models.build import BuildModel
from torch import nn

import torch
from models.containers import ModuleList,Module
from torch.nn import functional as F

def build_col_left_one():
    en = Encoder_Col(3,0)
    de = Decoder_Col(10201,54,3,1)
    return Transformer(2,en,de)

BuildModel.add(1,build_col_left_one)

class MultiLevelEncoder_Col(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False):
        super(MultiLevelEncoder_Col, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        
        outs = []
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        # out = input
        # for l in self.layers:
        #     out = l(out, out, out, attention_mask, attention_weights)
        return outs, attention_mask

class Encoder_Col(MultiLevelEncoder_Col):
    def __init__(self, N, padding_idx, d_in=2048):
        super(Encoder_Col, self).__init__(N, padding_idx)
        
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        return super(Encoder_Col, self).forward(out, attention_weights=attention_weights)


class Decoder_Col(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(Decoder_Col, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)
    
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output[:,i], mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)








