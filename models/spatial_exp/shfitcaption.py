import imp
import torch
import os
import torch.nn.functional as F
from torch import nn
from typing import List, Optional

from .attention import MultiHeadAttention
from models.containers import ModuleList, Module
from models.transformer.utils import PositionWiseFeedForward,generate_ref_points,sinusoid_encoding_table

from .shiftatt import ShiftHeadAttention


class QueryLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(QueryLayer, self).__init__()

        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, pos_enc=None, pos_emb=None):
        #一般情况下attention_mask为None, pos是detr的query的位置编码

        if pos_emb is not None:
            keys = keys + pos_emb
            values = values + pos_emb
        
        if pos_enc is not None:
            queries = queries + pos_enc

        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)

        ff = self.pwff(att)
        return ff

class ShiftLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,  k=4, 
                             scales=1, identity_map_reordering=False):
        super(ShiftLayer, self).__init__()

        self.mhatt = ShiftHeadAttention(h=h, d_model=d_model, k=k, scales=scales, dropout=dropout)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src_tensors, ref_points, src_masks, src_key_padding_masks, poses, src_querys):
        
        #一般为None
        if src_masks is None:
            src_masks =[None] * len(src_tensors)
        
        if src_key_padding_masks is None:
            src_key_padding_masks = [None] * len(src_tensors)

        if poses is None:
            poses = [None] * len(src_tensors)

        outs = []
        src_tensors = [self.with_pos_embed(tensor, pos) for tensor, pos in zip(src_tensors, poses)]

        for src, ref_point, _, src_key_padding_mask, pos, src_query in zip(src_tensors,
                                                                ref_points,
                                                                src_masks,
                                                                src_key_padding_masks,
                                                                poses,
                                                                src_querys):
            out = self.mhatt(src, src_tensors, ref_point,query_mask=src_key_padding_mask, key_masks=src_key_padding_masks, src_query=src_query)
            src = src + self.dropout(out)
            src = self.norm(src)
            src = self.pwff(src)
            outs.append(src)

        if len(outs) == 1:
            return outs[0]

        return outs

class Encoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, d_in=2048,
                 identity_map_reordering=False, k=8, feat_height=7, feat_width=7, scales=1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.dropout = dropout

        self.offset_layers = nn.ModuleList([QueryLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                   identity_map_reordering=identity_map_reordering)
                                  for _ in range(N)])

        self.att_layers = nn.ModuleList([ShiftLayer(h=h, d_model=d_model, d_k=d_k, k=k, 
                                                 scales=scales,dropout=dropout) for _ in range(N)])
        
        self.N = N
        self.padding_idx = padding_idx
        self.feat_height = feat_height
        self.feat_width = feat_width
        self.k = k
        self.scales = scales

        
    def forward(self, input, attention_mask=None, pos=None, attention_weights=None, box_output=None, pos_emb = None):
        bs = input.shape[0]
        
        ref_point = generate_ref_points(self.feat_width,self.feat_height)
        
        out = input

        for i in range(self.N):
            offset_query = self.offset_layers[i](out, box_output, box_output,  attention_mask=None, attention_weights = attention_weights, pos_emb=pos_emb, pos_enc=pos)

            src = input.view(bs, self.feat_height, self.feat_width, -1)
            query_ = offset_query.view(bs, self.feat_height, self.feat_width, -1)
            mask_ = attention_mask.squeeze(1).squeeze(1).view(bs, self.feat_height, self.feat_width)
            pos_ = pos.view(bs, self.feat_height, self.feat_width, -1)
            ref_points = [ref_point,]

            for j in range(len(ref_points)):
                ref_points[j] = ref_points[j].type_as(input)
                ref_points[j] = ref_points[j].unsqueeze(0).repeat(bs, 1, 1, 1)

            src_tensors = [src, ]
            src_masks = [mask_, ]
            poses = [pos_, ]
            src_querys = [query_, ]
            
            out = self.att_layers[i](src_tensors, ref_points, src_masks=None, src_key_padding_masks=src_masks, poses=poses, src_querys=src_querys)
            out = out.view(bs, self.feat_height * self.feat_width, -1)

        return out

class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1 , incor = 'concat'):
        super(DecoderLayer, self).__init__()

        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)
        # self.box_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)

        # self.dropout3 = nn.Dropout(dropout)
        # self.lnorm3 = nn.LayerNorm(d_model)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.cross_ = incor

    def forward(self, input, enc_output, box_lastlayer, mask_pad, mask_self_att, mask_enc_att):

        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad


        cross_ = enc_output
        if self.cross_ == 'concat':
            cross_ = torch.concat([enc_output,box_lastlayer],dim=-2)
            box_mask = (torch.sum(box_lastlayer, -1) == 0).unsqueeze(1).unsqueeze(1)

            mask_enc_att = torch.concat([mask_enc_att, box_mask],dim = -1)


        enc_att = self.enc_att(self_att, cross_, cross_,mask_enc_att)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        enc_att = enc_att * mask_pad


        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff

class Decoder(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                 dropout=.1, incor = 'concat'):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, incor = incor) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, box_lastlayer, mask_encoder):

        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        if self._is_stateful:
            self.running_mask_self_attention = torch.cat(
                [self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)

        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)

        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, box_lastlayer, mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

