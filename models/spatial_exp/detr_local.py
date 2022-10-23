import torch
import os
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List

#model
from .criterion import SetCriterion
from .postprocess import PostProcess
#utils
from .matcher import build_matcher

from models.transformer import MLP, MultiHeadAttention, PositionWiseFeedForward
from .position_encoding import build_position_encoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(EncoderLayer, self).__init__()

        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, pos=None):
        #pos: 这是二维坐标位置编码
        if pos is not None:
            queries = queries + pos
            keys = keys + pos
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)

        ff = self.pwff(att)
        return ff


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(DecoderLayer, self).__init__()

        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)
        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, object_query, enc_output, mask_self_att=None, mask_enc_att=None, pos=None,
                pos_emb=None):

        if pos_emb is not None:
            q = k = object_query + pos_emb
            # MHA+AddNorm
        self_att = self.self_att(q, k, object_query, mask_self_att)
        self_att = self.lnorm1(object_query + self.dropout1(self_att))

        # MHA+AddNorm
        if pos_emb is not None:
            self_att = self_att + pos_emb

        enc_att = self.enc_att(self_att, enc_output + pos, enc_output, mask_enc_att)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        # FFN+AddNorm

        ff = self.pwff(enc_att)

        return ff

class Encoder(nn.Module):
    def __init__(self, N,  d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, 
                 identity_map_reordering=False, norm=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                   identity_map_reordering=identity_map_reordering)
                                  for _ in range(N)])
        self.N = N

        self.norm = norm


    def forward(self, input, attention_mask=None, attention_weights=None, pos_grid=None):

        out = input

        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights, pos=pos_grid)

        if self.norm is not None:
            out = self.norm(out)

        return out

class Decoder(nn.Module):
    def __init__(self, N_dec, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, aux_outputs=False, norm=None):
        super(Decoder, self).__init__()

        self.norm = norm
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_dec)])

        self.N = N_dec
        self.aux_outputs = aux_outputs
        

    def forward(self, tgt, encoder_output, mask_enc, pos_grid, pos_emb):

        output = tgt

        intermediate = []
        for i, l in enumerate(self.layers):
            output = l(output, encoder_output, mask_enc_att=mask_enc, pos=pos_grid, pos_emb=pos_emb)
            if self.norm:
                output = self.norm(output)
            if self.aux_outputs:
                intermediate.append(output)

        if self.aux_outputs:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class Detr_Transformer(nn.Module):
    def __init__(self, d_model=512, h=8, num_enc=6, num_dec=6, d_ff=2048, dropout=0.1,
                    num_classes=1601, aux_outputs=False, num_queries=100, norm =True, num_channels = 2048):
        super(Detr_Transformer,self).__init__()

        self.d_model = d_model
        self.nhead = h

        self.norm = None

        if norm:
            self.norm = nn.LayerNorm(d_model)

        self.encoder = Encoder(N_dec=num_enc, norm=self.norm, d_model=d_model, h=h, d_ff=d_ff, dropout=dropout)
        self.decoder = Decoder(N_dec=num_dec, norm=self.norm, d_model=d_model, dropout=dropout, d_ff=d_ff, h=h, 
                                aux_outputs=aux_outputs)

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.aux_outputs = aux_outputs

        self.input_proj = nn.Conv2d(num_channels, d_model, kernel_size=1)

        self.build_loss()

        self.postProcess = PostProcess()

        self.pos_enc = build_position_encoding(512, 'sine')

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def forward(self, features, masks):

        bs = features.shape[0]

        input = self.input_proj(features).permute(0, 2, 3, 1).contiguous().view(bs, -1, self.d_model)#输入

        pos_grid = self.pos_enc(features, masks).permute(0, 2, 3, 1).contiguous().view(bs, -1, self.d_model)

        mask_enc = masks.view(bs, -1).unsqueeze(1).unsqueeze(1)

        pos_emb = self.query_embed.weight
        pos_emb = pos_emb.unsqueeze(0).repeat(bs, 1, 1)

        tgt = torch.zeros_like(pos_emb)

        memory = self.encoder(input, attention_mask=mask_enc, pos_grid=pos_grid)
        hs = self.decoder(tgt, memory, mask_enc, pos_grid=pos_grid, pos_emb=pos_emb)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_outputs:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out, hs[-1], pos_emb


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):

        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


    def build_loss(self):

        self.matcher = build_matcher()
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5}  # 1 ，5
        weight_dict['loss_giou'] = 2

        if self.aux_outputs:
            aux_weight_dict = {}
            for i in range(self.decoder.N - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ['labels', 'boxes', 'cardinality']

        self.criterion = SetCriterion(self.num_classes, matcher=self.matcher, weight_dict=weight_dict,
                                      eos_coef=0.1, losses=losses)


    def forward_box_loss(self, output, target):

        loss_dict = self.criterion(output, target)
        weight_dict = self.criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        return losses


    def dump_dt(self,output,ids,sizes):
        path_root = os.getcwd()
        results = self.postProcess(output, sizes)

        for i,id in enumerate(ids):
            with open(os.path.join(path_root,'input','detection-results','%s.txt'%id), 'w') as f:
                result = results[i]
                s = result['scores']
                l = result['labels']
                b = result['boxes']
                for k in range(100):
                    s1 = s[k]
                    l1 = l[k]
                    x1,y1,x2,y2 = b[k]
                    f.write(str(int(l1))+' '+str(float(s1))+' '+str(float(x1))+' '+str(float(y1))+' '+str(float(x2))+' '+str(float(y2))+'\n')


    def dump_gt(self,target,boxfeild):
        path_root = os.getcwd()
        sizes = []
        ids = []
        for t in target:
            id = t['id']
            with open(os.path.join(path_root,'input','ground-truth','%s.txt'%id), 'w') as f:
                boxes = t['boxes']
                labels = t['labels']
                boxes,size= boxfeild.xyxy_to_xywh(id,boxes)
                sizes.append(size)
                for label,box in zip(labels,boxes):
                    x1,y1,x2,y2 = box
                    f.write(str(int(label))+' '+str(float(x1))+' '+str(float(y1))+' '+str(float(x2))+' '+str(float(y2))+'\n')
            ids.append(id)
        return ids,sizes