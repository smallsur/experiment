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

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu

    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class EncoderLayer_Box(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=.1):
        super(EncoderLayer_Box, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, h, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn('relu')


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, src, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, 
                    pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DecoderLayer_Box(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=.1):
        super(DecoderLayer_Box, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, h, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, h, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn('relu')



    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Encoder_Box(nn.Module):
    def __init__(self, N_dec, norm = None, d_model=512, h=8, d_ff=2048, dropout=.1):
        super(Encoder_Box, self).__init__()

        self.layers = nn.ModuleList([EncoderLayer_Box(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout) for _ in range(N_dec)])

        self.norm = norm

        self.N = N_dec


    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):

        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Decoder_Box(nn.Module):
    def __init__(self, N_dec, norm=None, d_model=512, h=8, d_ff=2048, dropout=.1, return_inter = False):

        super(Decoder_Box, self).__init__()

        self.d_model = d_model
        self.norm = norm

        self.layers = nn.ModuleList([DecoderLayer_Box(d_model=d_model, h=h, 
                                        d_ff=d_ff, dropout=dropout) for _ in range(N_dec)])
        
        self.N = N_dec
        self.return_intermediate = return_inter

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class Detr_Transformer(nn.Module):
    def __init__(self, d_model=512, h=8, num_enc=6, num_dec=6, d_ff=2048, dropout=0.1,
                    num_classes=1601, aux_outputs=False, num_queries=100, norm =True, num_channels = 2048):
        super().__init__()

        self.d_model = d_model
        self.nhead = h

        self.norm = None
        if norm:
            self.norm = nn.LayerNorm(d_model)

        self.encoder = Encoder_Box(N_dec=num_enc, norm=self.norm, d_model=d_model, h=h, d_ff=d_ff, dropout=dropout)
        self.decoder = Decoder_Box(N_dec=num_dec, norm=self.norm, return_inter=aux_outputs, d_model=d_model, dropout=dropout, 
                            d_ff=d_ff,h = h)

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.aux_outputs = aux_outputs

        self.input_proj = nn.Conv2d(num_channels, d_model, kernel_size=1)

        self.build_loss()

        self.postProcess = PostProcess()

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def forward(self,features, masks, poses):
        
        assert len(features) == 1
        src = features[-1]
        pos_embed = poses[-1]

        src = self.input_proj(src)
        src = src.flatten(2).permute(2, 0, 1)#805*2*256

        bs, c, h, w = src.shape
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        query_emb = self.query_embed.weight
        query_emb = query_emb.unsqueeze(1).repeat(1, bs, 1)
        
        masks = masks.flatten(1)

        tgt = torch.zeros_like(query_emb)

        memory = self.encoder(src, src_key_padding_mask=masks, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=masks,
                          pos=pos_embed, query_pos=query_emb)

        hs = hs.transpose(1, 2)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_outputs:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out, hs, query_emb


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
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



