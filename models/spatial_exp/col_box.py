from re import I
import torch
import torch.nn.functional as F
from torch import nn



from .attention import MultiHeadAttention
from .position_encoding import build_position_encoding
from models.transformer.utils import sinusoid_encoding_table
from models.containers import ModuleList,Module
from models.build import BuildModel
from models.transformer.utils import PositionWiseFeedForward
from models.captioning_model import CaptioningModel
from .matcher import build_matcher
from utils import box_cxcywh_to_xyxy, generalized_box_iou,is_dist_avail_and_initialized,get_world_size,accuracy

num_classes = 1601

#  args.hidden_dim,args.lr_backbone,
def col_box(args):
    encoder = MultiLevelEncoder_Box(6,padding_idx=0)
    decoder_box = Decoder_Box(6)
    decoder_cap = Decoder_Caption(10201,54,3,1)
    return Transformer(2,encoder,decoder_box,decoder_cap)

BuildModel.add(3,col_box)



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

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

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None,pos=None):
        if pos is not None:
            queries = queries + pos
            keys = keys + pos
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)

        ff = self.pwff(att)
        return ff



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

    def forward(self, object_query, enc_output, mask_pad =None, mask_self_att=None, mask_enc_att=None,pos=None,pos_emb=None):
        if pos_emb is not None:
            q = k = object_query + pos_emb 
        # MHA+AddNorm
        self_att = self.self_att(q, k, object_query, mask_self_att)
        self_att = self.lnorm1(object_query + self.dropout1(self_att))

        # MHA+AddNorm
        if pos_emb is not None:
            self_att = self_att + pos_emb

        enc_att = self.enc_att(self_att, enc_output + pos , enc_output, mask_enc_att)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        # FFN+AddNorm

        ff = self.pwff(enc_att)

        return ff


class DecoderLayer_Caption(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(DecoderLayer_Caption, self).__init__()
        
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False)
        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        # MHA+AddNorm
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad
        # MHA+AddNorm
        enc_att = self.enc_att(self_att, enc_output, enc_output, mask_enc_att)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        enc_att = enc_att * mask_pad
        # FFN+AddNorm
        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff



class MultiLevelEncoder_Box(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,d_in=2048,
                 identity_map_reordering=False):
        super(MultiLevelEncoder_Box, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = ModuleList([EncoderLayer_Box(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering)
                                     for _ in range(N)])
        self.pos = build_position_encoding(512,'sine')
        self.padding_idx = padding_idx
        self.fc = nn.Linear(d_in, self.d_model)
        self.drop = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):

        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        pos_ = self.pos(input,attention_mask)

        outs = []
        out = input

        out = F.relu(self.fc(input))
        out = self.drop(out)
        out = self.layer_norm(out)

        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights,pos=pos_)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        # out = input
        # for l in self.layers:
        #     out = l(out, out, out, attention_mask, attention_weights)
        return out, attention_mask, pos_



class Decoder_Box(Module):
    def __init__(self, N_dec, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, num_queries=100,num_classes=1601,aux_outputs = False):
        super(Decoder_Box, self).__init__()

        self.d_model = d_model

        self.layers = ModuleList(
            [DecoderLayer_Box(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_dec)])

        self.N = N_dec
        self.num_classes = num_classes
        self.num_queries =num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)################################

        self.aux_outputs = aux_outputs


    def forward(self, encoder_output, mask_encoder , pos):

        bs = pos.shape[0]
        pos_emb = self.query_embed.weight
        pos_emb = pos_emb.unsqueeze(0).repeat(bs, 1, 1)
        tgt = torch.zeros_like(pos_emb)

        intermediate = []
        for i, l in enumerate(self.layers):
            tgt = l(tgt,encoder_output,mask_enc_att=mask_encoder, pos=pos,pos_emb=pos_emb)
            intermediate.append(tgt)


        intermediate = torch.stack(intermediate)

        outputs_class = self.class_embed(intermediate)
        outputs_coord = self.bbox_embed(intermediate).sigmoid()

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.aux_outputs:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]



class Decoder_Caption(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(Decoder_Caption, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer_Caption(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N_dec)])
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
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)


class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder_box ,decoder_cap):
        super(Transformer, self).__init__()

        self.bos_idx = bos_idx

        self.encoder = encoder
        self.decoder_box = decoder_box
        self.decoder_cap = decoder_cap


        self.num_classes =self.decoder_box.num_classes
        self.aux_outputs = self.decoder_box.aux_outputs

    
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        # self.register_state('cap_output',None)
        self.register_state('pos',None)

        self.build_loss()

        self.init_weights()

    def build_loss(self):
        self.matcher = build_matcher()
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5} #1 ，5
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

    def forward(self, images, seq):
        enc_output, mask_enc, pos= self.encoder(images)
        # dec_output = self.decoder(seq, enc_output, mask_enc)
        box_output = self.decoder_box(enc_output,mask_enc,pos)
        cap_output = self.decoder_cap(seq, enc_output, mask_enc)
        return box_output,cap_output

    def forward_box_loss(self,output,target):
        loss_dict = self.criterion(output,target)
        weight_dict = self.criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        return losses

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.enc_output, self.mask_enc,self.pos = self.encoder(visual)
                
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder_cap(it, self.enc_output, self.mask_enc)
    



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
        src_logits = outputs['pred_logits']#2*100*92

        idx = self._get_src_permutation_idx(indices)#batchidx,idx
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])#batch*numbox
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











