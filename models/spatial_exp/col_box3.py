from re import I
import torch
import os
import torch.nn.functional as F
from torch import nn
from typing import List, Optional

from .attention import MultiHeadAttention
from .position_encoding import build_position_encoding
from models.transformer.utils import sinusoid_encoding_table
from models.containers import ModuleList, Module
from models.build import BuildModel
from models.transformer.utils import PositionWiseFeedForward
from models.captioning_model import CaptioningModel
from .matcher import build_matcher
from utils import box_cxcywh_to_xyxy, generalized_box_iou, is_dist_avail_and_initialized, get_world_size, accuracy
from .evalue_box import evalue_box


num_classes = 1601


#  args.hidden_dim,args.lr_backbone,
def col_box_3(args):
    encoder_box = MultiLevelEncoder_Box(3, padding_idx=0)
    encoder_cap = MultiLevelEncoder_Cap(3, padding_idx=0)

    decoder_box = Decoder_Box(3, aux_outputs=args.aux_outputs)
    decoder_cap = Decoder_Caption(10201, 54, 3, 1, incor= 'single')
    return Transformer(2, encoder_box, encoder_cap, decoder_box, decoder_cap)


BuildModel.add(5, col_box_3)


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
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(EncoderLayer_Box, self).__init__()

        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, pos=None):
        if pos is not None:
            queries = queries + pos
            keys = keys + pos
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)

        ff = self.pwff(att)
        return ff

class EncoderLayer_Cap(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(EncoderLayer_Cap, self).__init__()

        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, pos=None):
        if pos is not None:
            queries = queries + pos
            keys = keys + pos
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)

        ff = self.pwff(att)
        return ff

class Layer_Offset(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(Layer_Offset, self).__init__()

        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, pos=None):

        if pos is not None:
            keys = keys + pos
            values = values + pos

        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)

        ff = self.pwff(att)
        return ff

class Layer_Incor_offset(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,  k=4, 
                            last_feat_height=7, last_feat_width=7, scales=1, identity_map_reordering=False):
        super(Layer_Incor_offset, self).__init__()

        self.mhatt = ShiftHeadAttention(h=h, d_model=d_model, k=k, 
                                                last_feat_height=last_feat_height, last_feat_width=last_feat_width,
                                                scales=scales,dropout=dropout)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src_tensors, ref_points, src_masks, src_key_padding_masks, poses, src_querys):
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
            out, _ = self.mhatt(src, src_tensors, ref_point,query_mask=src_key_padding_mask, key_masks=src_key_padding_masks, src_query=src_query)
            out = self.pwff(out)
            outs.append(out)

        if len(outs) == 1:
            return outs[0]

        return outs

class MultiLevelEncoder_Box(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, d_in=2048,
                 identity_map_reordering=False):
        super(MultiLevelEncoder_Box, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = ModuleList([EncoderLayer_Box(d_model, d_k, d_v, h, d_ff, dropout,
                                                   identity_map_reordering=identity_map_reordering)
                                  for _ in range(N)])

        self.padding_idx = padding_idx

    def forward(self, input, attention_mask=None, pos_=None, attention_weights=None):

        outs = []
        out = input

        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights, pos=pos_)
            outs.append(out.unsqueeze(1))

        return out

class MultiLevelEncoder_Cap(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, d_in=2048,
                 identity_map_reordering=False, k=4, feat_height=7, feat_width=7, scales=1):
        super(MultiLevelEncoder_Cap, self).__init__()

        self.d_model = d_model
        self.dropout = dropout

        self.cap_layers = ModuleList([EncoderLayer_Box(d_model, d_k, d_v, h, d_ff, dropout,
                                                   identity_map_reordering=identity_map_reordering)
                                  for _ in range(N)])

        self.offset_layers = ModuleList([Layer_Offset(d_model, d_k, d_v, h, d_ff, dropout,
                                                   identity_map_reordering=identity_map_reordering)
                                  for _ in range(N)])

        self.att_layers = ModuleList([Layer_Incor_offset(h=h, d_model=d_model, d_k=d_k, k=k, 
                                                last_feat_height=feat_height, last_feat_width=feat_width, scales=scales,dropout=dropout)])
        
        self.N = N
        self.padding_idx = padding_idx
        self.feat_height = feat_height
        self.feat_width = feat_width
        self.k = k
        self.scales = scales

        

    def forward(self, input, attention_mask=None, pos=None, attention_weights=None, box_output=None, pos_emb = None):
        bs = input.shape[0]
        
        ref_point = generate_ref_points(self.feat_width,self.feat_height)
        
        for l in self.cap_layers:
            input = l(input,input,input,attention_mask, attention_weights, pos=pos)

        out = input 

        for i in range(self.N):
            offset_query = self.offset_layers[i](out, box_output, box_output,  attention_mask=None, attention_weights = attention_weights, pos=pos_emb)
            # out = torch.masked_fill(out, attention_mask, 0)

            src = input.view(bs, self.feat_height, self.feat_width, -1)
            query_ = offset_query.view(bs, self.feat_height, self.feat_width, -1)
            mask_ = attention_mask.squeeze(1).squeeze(1).view(bs, self.feat_height, self.feat_width)
            pos_ = pos.view(bs, self.feat_height, self.feat_width, -1)

            ref_points = [ref_point,]
            for i in range(len(ref_points)):
                ref_points[i] = ref_points[i].type_as(input)
                ref_points[i] = ref_points[i].unsqueeze(0).repeat(bs, 1, 1, 1)

            src_tensors = [src, ]
            src_masks = [mask_, ]
            poses = [pos_, ]
            src_querys = [query_, ]
            
            out = self.att_layers[i](src_tensors, ref_points, src_masks=None, src_key_padding_masks=src_masks, poses=poses, src_querys=src_querys)
            out = out.view(bs, self.feat_height * self.feat_width, -1)

        return out


class DecoderLayer_Box(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(DecoderLayer_Box, self).__init__()

        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)
        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, object_query, enc_output, mask_pad=None, mask_self_att=None, mask_enc_att=None, pos=None,
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


class DecoderLayer_Caption(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1 , incor = 'concat'):
        super(DecoderLayer_Caption, self).__init__()

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

        # enc_att = self.enc_att(self_att, enc_output, enc_output, mask_enc_att)

        cross_ = enc_output
        if self.cross_ == 'concat':
            cross_ = torch.concat([enc_output,box_lastlayer],dim=-2)
            box_mask = (torch.sum(box_lastlayer, -1) == 0).unsqueeze(1).unsqueeze(1)

            mask_enc_att = torch.concat([mask_enc_att, box_mask],dim = -1)


        enc_att = self.enc_att(self_att, cross_, cross_,mask_enc_att)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        enc_att = enc_att * mask_pad

        # box_att = self.box_att(enc_att,box_lastlayer,box_lastlayer)
        # box_att = self.lnorm3(enc_att + self.dropout3(box_att))
        # box_att = box_att * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class Decoder_Box(Module):
    def __init__(self, N_dec, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, num_queries=100,
                 num_classes=1601, aux_outputs=False):
        super(Decoder_Box, self).__init__()

        self.d_model = d_model

        self.layers = ModuleList(
            [DecoderLayer_Box(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_dec)])

        self.N = N_dec
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)  

        self.aux_outputs = aux_outputs

    def forward(self, encoder_output, mask_encoder, pos):

        bs = pos.shape[0]
        pos_emb = self.query_embed.weight
        pos_emb = pos_emb.unsqueeze(0).repeat(bs, 1, 1)
        tgt = torch.zeros_like(pos_emb)

        intermediate = []
        for i, l in enumerate(self.layers):
            tgt = l(tgt, encoder_output, mask_enc_att=mask_encoder, pos=pos, pos_emb=pos_emb)
            intermediate.append(tgt)

        intermediate = torch.stack(intermediate)

        outputs_class = self.class_embed(intermediate)
        outputs_coord = self.bbox_embed(intermediate).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.aux_outputs:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out, tgt, pos_emb

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class Decoder_Caption(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                 dropout=.1, incor = 'concat'):
        super(Decoder_Caption, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer_Caption(d_model, d_k, d_v, h, d_ff, dropout, incor = incor) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, box_lastlayer, mask_encoder):
        # input (b_s, seq_len)
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


class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder_box, encoder_cap, decoder_box, decoder_cap, d_in=2048):
        super(Transformer, self).__init__()

        self.bos_idx = bos_idx

        self.encoder_box = encoder_box
        self.encoder_cap = encoder_cap

        self.decoder_box = decoder_box
        self.decoder_cap = decoder_cap

        self.num_classes = self.decoder_box.num_classes
        self.aux_outputs = self.decoder_box.aux_outputs

        self.padding_idx = self.encoder_box.padding_idx

        self.register_state('enc_box_output', None)
        self.register_state('enc_cap_output', None)

        self.register_state('mask_enc', None)

        self.register_state('pos', None)
        self.register_state('box_lastlayer',None)

        self.register_state('input',None)

        self.build_loss()

        self.postProcess = PostProcess()

        self.pos_enc = build_position_encoding(512, 'sine')

        self.fc = nn.Linear(d_in, self.encoder_box.d_model)
        self.drop = nn.Dropout(p=self.encoder_box.dropout)
        self.layer_norm = nn.LayerNorm(self.encoder_box.d_model)

        self.init_weights()

    def build_loss(self):
        self.matcher = build_matcher()
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5}  # 1 ，5
        weight_dict['loss_giou'] = 2

        if self.aux_outputs:
            aux_weight_dict = {}
            for i in range(self.decoder_box.N - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ['labels', 'boxes', 'cardinality']

        self.criterion = SetCriterion(self.num_classes, matcher=self.matcher, weight_dict=weight_dict,
                                      eos_coef=0.1, losses=losses)

    @property
    def d_model(self):
        return self.decoder_box.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, only_box = False):
        # prepare data
        input, mask_enc, pos_ = self.forward_input(images)

        #into encoder_box
        enc_box_output = self.encoder_box(input, mask_enc, pos_)

        #generate boxinfo
        box_output, box_lastlayer, pos_emb = self.decoder_box(enc_box_output, mask_enc, pos_)

        
        if only_box:
            return box_output

        #into encoder_cap
        enc_cap_output = self.encoder_cap(input, mask_enc, pos_, attention_weights=None, box_output=box_lastlayer, pos_emb=pos_emb)

        #generate caption
        cap_output = self.decoder_cap(seq, enc_cap_output, box_lastlayer, mask_enc)

        return box_output, cap_output

    def forward_box_loss(self, output, target):
        loss_dict = self.criterion(output, target)
        weight_dict = self.criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        return losses

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]
    
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
        

    def evalue_box_(self,args):
        return evalue_box(args)
    
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
                

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.input, self.mask_enc, self.pos = self.forward_input(visual)
                self.enc_box_output = self.encoder_box(self.input, self.mask_enc, self.pos)
                _, self.box_lastlayer, pos_emb = self.decoder_box(self.enc_box_output, self.mask_enc, self.pos)


                self.enc_cap_output = self.encoder_cap(self.input, self.mask_enc, self.pos, attention_weights=None, box_output=self.box_lastlayer, pos_emb=pos_emb)
                
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder_cap(it, self.enc_cap_output, self.box_lastlayer, self.mask_enc)

    def forward_input(self, input):
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        pos_ = self.pos_enc(input, attention_mask)

        out = input

        out = F.relu(self.fc(input))
        out = self.drop(out)
        out = self.layer_norm(out)

        return out, attention_mask, pos_


class SetCriterion(nn.Module):


    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # 2*100*92

        idx = self._get_src_permutation_idx(indices)  # batchidx,idx
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # batch*numbox
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):

    @torch.no_grad()
    def forward(self, outputs, target_sizes):

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        target_sizes = torch.tensor(target_sizes, dtype=torch.long,device=out_logits.device)
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


def generate_ref_points(width: int,
                        height: int):
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    grid_y = grid_y / (height - 1)
    grid_x = grid_x / (width - 1)

    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    return grid


def restore_scale(width: int,
                  height: int,
                  ref_point: torch.Tensor):
    new_point = ref_point.clone().detach()
    new_point[..., 0] = new_point[..., 0] * (width - 1)
    new_point[..., 1] = new_point[..., 1] * (height - 1)

    return new_point


class ShiftHeadAttention(nn.Module):
    def __init__(self, h,
                 d_model,
                 k,
                 last_feat_height,
                 last_feat_width,
                 scales=1,
                 dropout=0.1,
                 need_attn=False):
        """
        :param h: number of self attention head
        :param d_model: dimension of model
        :param dropout:
        :param k: number of keys
        """
        super(ShiftHeadAttention, self).__init__()
        assert h == 8  # currently header is fixed 8 in paper
        assert d_model % h == 0
        # We assume d_v always equals d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(d_model / h)
        self.h = h

        self.q_proj = nn.Linear(d_model, d_model)
        self.b_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        self.scales_hw = []
        for i in range(scales):
            self.scales_hw.append([last_feat_height * 2 ** i,
                                   last_feat_width * 2 ** i])

        self.dropout = None

        if self.dropout:
            self.dropout = nn.Dropout(p=dropout)

        self.k = k
        self.scales = scales
        self.last_feat_height = last_feat_height
        self.last_feat_width = last_feat_width

        self.offset_dims = 2 * self.h * self.k * self.scales
        self.A_dims = self.h * self.k * self.scales

        # 2MLK for offsets MLK for A_mlqk
        self.offset_proj = nn.Linear(d_model, self.offset_dims)
        self.A_proj = nn.Linear(d_model, self.A_dims)

        self.wm_proj = nn.Linear(d_model, d_model)
        self.need_attn = need_attn
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.offset_proj.weight, 0.0)
        torch.nn.init.constant_(self.A_proj.weight, 0.0)

        torch.nn.init.constant_(self.A_proj.bias, 1 / (self.scales * self.k))

        def init_xy(bias, x, y):
            torch.nn.init.constant_(bias[:, 0], float(x))
            torch.nn.init.constant_(bias[:, 1], float(y))

        # caution: offset layout will be  M, L, K, 2
        bias = self.offset_proj.bias.view(self.h, self.scales, self.k, 2)

        init_xy(bias[0], x=-self.k, y=-self.k)
        init_xy(bias[1], x=-self.k, y=0)
        init_xy(bias[2], x=-self.k, y=self.k)
        init_xy(bias[3], x=0, y=-self.k)
        init_xy(bias[4], x=0, y=self.k)
        init_xy(bias[5], x=self.k, y=-self.k)
        init_xy(bias[6], x=self.k, y=0)
        init_xy(bias[7], x=self.k, y=self.k)

    def forward(self,
                query: torch.Tensor,
                keys: List[torch.Tensor],
                ref_point: torch.Tensor,
                query_mask: torch.Tensor = None,
                key_masks: Optional[torch.Tensor] = None,
                src_query = None):
        """
        :param key_masks:
        :param query_mask:
        :param query: B, H, W, C
        :param keys: List[B, H, W, C]
        :param ref_point: B, H, W, 2
        :return:
        """
        if key_masks is None:
            key_masks = [None] * len(keys)

        assert len(keys) == self.scales

        attns = {'attns': None, 'offsets': None}

        nbatches, query_height, query_width, _ = query.shape

        # B, H, W, C
        query = self.q_proj(query)
        #
        src_query = self.b_proj(src_query)

        # B, H, W, 2MLK
        offset = self.offset_proj(src_query)
        # B, H, W, M, 2LK
        offset = offset.view(nbatches, query_height, query_width, self.h, -1)

        # B, H, W, MLK
        A = self.A_proj(src_query)

        # B, H, W, 1, mask before softmax
        if query_mask is not None:
            query_mask_ = query_mask.unsqueeze(dim=-1)
            _, _, _, mlk = A.shape
            query_mask_ = query_mask_.expand(nbatches, query_height, query_width, mlk)
            A = torch.masked_fill(A, mask=query_mask_, value=float('-inf'))

        # B, H, W, M, LK
        A = A.view(nbatches, query_height, query_width, self.h, -1)
        A = F.softmax(A, dim=-1)

        # mask nan position
        if query_mask is not None:
            # B, H, W, 1, 1
            query_mask_ = query_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            A = torch.masked_fill(A, query_mask_.expand_as(A), 0.0)

        if self.need_attn:
            attns['attns'] = A
            attns['offsets'] = offset

        offset = offset.view(nbatches, query_height, query_width, self.h, self.scales, self.k, 2)
        offset = offset.permute(0, 3, 4, 5, 1, 2, 6).contiguous()
        # B*M, L, K, H, W, 2
        offset = offset.view(nbatches * self.h, self.scales, self.k, query_height, query_width, 2)

        A = A.permute(0, 3, 1, 2, 4).contiguous()
        # B*M, H*W, LK
        A = A.view(nbatches * self.h, query_height * query_width, -1)

        scale_features = []
        for l in range(self.scales):
            feat_map = keys[l]
            _, h, w, _ = feat_map.shape

            key_mask = key_masks[l]

            # B, H, W, 2
            reversed_ref_point = restore_scale(height=h, width=w, ref_point=ref_point)

            # B, H, W, 2 -> B*M, H, W, 2
            reversed_ref_point = reversed_ref_point.repeat(self.h, 1, 1, 1)

            # B, h, w, M, C_v
            scale_feature = self.k_proj(feat_map).view(nbatches, h, w, self.h, self.d_k)

            if key_mask is not None:
                # B, h, w, 1, 1
                key_mask = key_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
                key_mask = key_mask.expand(nbatches, h, w, self.h, self.d_k)
                scale_feature = torch.masked_fill(scale_feature, mask=key_mask, value=0)

            # B, M, C_v, h, w
            scale_feature = scale_feature.permute(0, 3, 4, 1, 2).contiguous()
            # B*M, C_v, h, w
            scale_feature = scale_feature.view(-1, self.d_k, h, w)

            k_features = []

            for k in range(self.k):
                points = reversed_ref_point + offset[:, l, k, :, :, :]
                vgrid_x = 2.0 * points[:, :, :, 0] / max(w - 1, 1) - 1.0
                vgrid_y = 2.0 * points[:, :, :, 1] / max(h - 1, 1) - 1.0
                vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

                # B*M, C_v, H, W
                feat = F.grid_sample(scale_feature, vgrid_scaled, mode='bilinear', padding_mode='zeros')
                k_features.append(feat)
            # B*M, k, C_v, H, W
            k_features = torch.stack(k_features, dim=1)
            scale_features.append(k_features)

        # B*M, L, K, C_v, H, W
        scale_features = torch.stack(scale_features, dim=1)

        # B*M, H*W, C_v, LK
        scale_features = scale_features.permute(0, 4, 5, 3, 1, 2).contiguous()
        scale_features = scale_features.view(nbatches * self.h, query_height * query_width, self.d_k, -1)

        # B*M, H*W, C_v
        feat = torch.einsum('nlds, nls -> nld', scale_features, A)

        # B*M, H*W, C_v -> B, M, H, W, C_v
        feat = feat.view(nbatches, self.h, query_height, query_width, self.d_k)
        # B, M, H, W, C_v -> B, H, W, M, C_v
        feat = feat.permute(0, 2, 3, 1, 4).contiguous()
        # B, H, W, M, C_v -> B, H, W, M * C_v
        feat = feat.view(nbatches, query_height, query_width, self.d_k * self.h)

        feat = self.wm_proj(feat)
        if self.dropout:
            feat = self.dropout(feat)

        return feat, attns


