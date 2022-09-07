from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.spatial_exp.attention import MultiHeadAttention

from models.containers import ModuleList


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(EncoderLayer, self).__init__()
        # self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering)
        # self.dropout = nn.Dropout(dropout)
        # self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        # att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        # outs = []
        # out = input
        # for l in self.layers:
        #     out = l(out, out, out, attention_mask, attention_weights)
        #     outs.append(out.unsqueeze(1))

        # outs = torch.cat(outs, 1)
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
        return out, attention_mask

class Encoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048):
        super(Encoder, self).__init__(N, padding_idx)
        
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        return super(Encoder, self).forward(out, attention_weights=attention_weights)