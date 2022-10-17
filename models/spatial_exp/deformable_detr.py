import torch
import os
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List

#model
from .attention import MultiHeadAttention
from detectron2.modeling.backbone import build_backbone
from torchvision.models._utils import IntermediateLayerGetter
from .deformable_att import DeformableHeadAttention
from .position_encoding import build_position_encoding, build_scale_embedding

#utils
from .deformable_att import generate_ref_points, restore_scale
from utils import box_cxcywh_to_xyxy, generalized_box_iou, is_dist_avail_and_initialized, get_world_size, accuracy
from .matcher import build_matcher

#config
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.config import CfgNode as CN


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
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=.1, k=4, 
                            last_feat_height=7, last_feat_width=7, scales=4):
        super(EncoderLayer_Box, self).__init__()

        self.mhatt = DeformableHeadAttention(h=h, d_model=d_model, k=k, 
                                                last_feat_height=last_feat_height, last_feat_width=last_feat_width,
                                                scales=scales, dropout=dropout)

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
    
    def forward(self,
                     src_tensors: List[Tensor],
                     ref_points: List[Tensor],
                     src_masks: Optional[List[Tensor]] = None,
                     src_key_padding_masks: Optional[List[Tensor]] = None,
                     poses: Optional[List[Tensor]] = None):

        if src_masks is None:
            src_masks = [None] * len(src_tensors)

        if src_key_padding_masks is None:
            src_key_padding_masks = [None] * len(src_tensors)

        if poses is None:
            poses = [None] * len(src_tensors)

        feats = []
        src_tensors = [self.with_pos_embed(tensor, pos) for tensor, pos in zip(src_tensors, poses)]
        for src, ref_point, _, src_key_padding_mask, pos in zip(src_tensors,
                                                                ref_points,
                                                                src_masks,
                                                                src_key_padding_masks,
                                                                poses):
            # src = self.with_pos_embed(src, pos)
            src2, _ = self.mhatt(src,
                                        src_tensors,
                                        ref_point,
                                        query_mask=src_key_padding_mask,
                                        key_masks=src_key_padding_masks)

            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            feats.append(src)

        return feats

class DecoderLayer_Box(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, k=4, 
                            last_feat_height=7, last_feat_width=7, scales=4):
        super(DecoderLayer_Box, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, h, dropout=dropout)
        self.ms_deformbale_attn = DeformableHeadAttention(h=h,
                                                          d_model=d_model,
                                                          k=k,
                                                          scales=scales,
                                                          last_feat_height=last_feat_height,
                                                          last_feat_width=last_feat_width,
                                                          dropout=dropout)
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
                     ref_point: Tensor,
                     tgt_mask: Optional[Tensor] = None,
                     memory_masks: Optional[List[Tensor]] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_masks: Optional[List[Tensor]] = None,
                     poses: Optional[List[Tensor]] = None,
                     query_pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        memory = [self.with_pos_embed(tensor, pos) for tensor, pos in zip(memory, poses)]

        # L, B, C -> B, L, 1, C
        tgt = tgt.transpose(0, 1).unsqueeze(dim=2)
        ref_point = ref_point.transpose(0, 1).unsqueeze(dim=2)

        # B, L, 1, C
        tgt2, attns = self.ms_deformbale_attn(tgt,
                                              memory,
                                              ref_point,
                                              query_mask=None,
                                              key_masks=memory_key_padding_masks)


        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # B, L, 1, C -> L, B, C
        tgt = tgt.squeeze(dim=2)
        tgt = tgt.transpose(0, 1).contiguous()

        # decoder we only one query tensor
        return tgt



class Encoder_Box(nn.Module):
    def __init__(self, N_dec, norm = None, d_model=512, h=8, d_ff=2048, dropout=.1, k=4, 
                            last_feat_height=7, last_feat_width=7, scales=4):
        super(Encoder_Box, self).__init__()

        self.layers = nn.ModuleList([EncoderLayer_Box(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout, k=k, 
                            last_feat_height=last_feat_height, last_feat_width=last_feat_width, scales=scales) for _ in range(N_dec)])

        self.norm = norm

        self.N = N_dec


    def forward(self,
                src_tensors: List[Tensor],
                ref_points: List[Tensor],
                src_masks: Optional[List[Tensor]] = None,
                src_key_padding_masks: Optional[List[Tensor]] = None,
                poses: Optional[List[Tensor]] = None):

        outputs = src_tensors

        for layer in self.layers:
            outputs = layer(outputs,
                            ref_points,
                            src_masks=src_masks,
                            src_key_padding_masks=src_key_padding_masks,
                            poses=poses)

        if self.norm is not None:
            for index, output in enumerate(outputs):
                outputs[index] = self.norm(output)

        return outputs

class Decoder_Box(nn.Module):
    def __init__(self, N_dec, norm=None, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, k=4, 
                    scales=4 ,last_feat_height = 7, last_feat_width = 7, return_inter = False):

        super(Decoder_Box, self).__init__()

        self.d_model = d_model
        self.norm = norm

        self.layers = nn.ModuleList([DecoderLayer_Box(d_model=d_model, d_k=d_k, d_v=d_v, h=h, 
                                        d_ff=d_ff, dropout=dropout, k=k, scales=scales, 
                                        last_feat_height=last_feat_height, last_feat_width=last_feat_width) for _ in range(N_dec)])
        
        self.N = N_dec
        self.return_intermediate = return_inter

    def forward(self, tgt, memory, ref_point: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_masks: Optional[List[Tensor]] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_masks: Optional[List[Tensor]] = None, poses: Optional[List[Tensor]] = None,
                query_pos: Optional[Tensor] = None):

        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output,
                           memory,
                           ref_point,
                           tgt_mask=tgt_mask,
                           memory_masks=memory_masks,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_masks=memory_key_padding_masks,
                           poses=poses, query_pos=query_pos)

            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output



class Detr_Transformer(nn.Module):
    def __init__(self, d_model=512, h=8, num_enc=6, num_dec=6, d_ff=2048, dropout=0.1,
                     scales=4, k=4, last_feat_height=16, last_feat_width=16, 
                    num_classes=1601, aux_outputs=False, num_queries=100, norm =True):
        super().__init__()

        self.d_model = d_model
        self.norm = None
        if norm:
            self.norm = nn.LayerNorm(d_model)

        self.encoder = Encoder_Box(N_dec=num_enc, norm=self.norm, d_model=d_model, h=h, d_ff=d_ff, dropout=dropout, k=k, 
                            last_feat_height=last_feat_height, last_feat_width=last_feat_width, scales = scales)
        self.decoder = Decoder_Box(N_dec=num_dec, norm=self.norm, return_inter=aux_outputs, d_model=d_model, dropout=dropout, 
                            k = k, scales=scales, last_feat_height=last_feat_height, last_feat_width=last_feat_width, d_ff=d_ff,
                            h = h)

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.aux_outputs = aux_outputs

        self.query_ref_point_proj = nn.Linear(d_model, 2)

        self.build_loss()

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def forward(self,features, masks, poses):

        bs = features[0].size(0)

        query_emb = self.query_embed.weight
        query_emb = query_emb.unsqueeze(1).repeat(1, bs, 1)
        
        for index in range(len(features)):
            features[index] = features[index].permute(0, 2, 3, 1)
            poses[index] = poses[index].permute(0, 2, 3, 1)
            # masks[index] = masks[index].permute(0, 2, 3, 1)

        ref_points = []
        for tensor in features:
            _, height, width, _ = tensor.shape
            ref_point = generate_ref_points(width=width,
                                            height=height)
            ref_point = ref_point.type_as(features[0])
            # H, W, 2 -> B, H, W, 2
            ref_point = ref_point.unsqueeze(0).repeat(bs, 1, 1, 1)
            ref_points.append(ref_point)

        tgt = torch.zeros_like(query_emb)
        
        memory = self.encoder(features, ref_points, src_key_padding_masks=masks, poses=poses)
        
        query_ref_point = self.query_ref_point_proj(tgt)
        query_ref_point = F.sigmoid(query_ref_point)

        hs = self.decoder(tgt, memory, query_ref_point, memory_key_padding_masks=masks,
                            poses=poses, query_pos=query_emb)

        hs = hs.transpose(1, 2).contiguous()

        ref_point = query_ref_point.transpose(0, 1).contiguous()

        outputs_class = self.class_embed(hs)

        inversed_ref_point = - torch.log(1 / (ref_point + 1e-10) - 1 + 1e-10)

        outputs_coord = self.bbox_embed(hs)

        outputs_coord[..., 0] = outputs_coord[..., 0] + inversed_ref_point[..., 0]
        outputs_coord[..., 1] = outputs_coord[..., 1] + inversed_ref_point[..., 1]

        outputs_coord = torch.sigmoid(outputs_coord)

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



class Backbone(nn.Module):

    def __init__(self, args, train_backbone=False, pixel_mean = [103.53, 116.28, 123.675], 
                    pixel_std = [57.375, 57.12, 58.395], d_model = 256, last_dim=2048):

        super(Backbone, self).__init__()

        self.backbone = build_backbone(self.setup(args)) # 72*96, 36*48, 18*24, 8*11

        self.path_ = os.path.join(args.dir_to_save_model, 'rs101.pth')


        for name, parameter in self.backbone.named_parameters():
            if not train_backbone or 'res3' not in name and 'res4' not in name and 'res5' not in name:
                parameter.requires_grad_(False)

        self.return_layers = {"res3": "0", "res4": "1", "res5": "2"}
        self.body = IntermediateLayerGetter(self.backbone, return_layers=self.return_layers)

        self.c3_conv = nn.Conv2d(last_dim//4, d_model, kernel_size=1)
        self.c4_conv = nn.Conv2d(last_dim//2, d_model, kernel_size=1)
        self.c5_conv = nn.Conv2d(last_dim, d_model, kernel_size=1)
        self.c6_conv = nn.Conv2d(last_dim, d_model, kernel_size=(3,3), stride=2)

        self.pos_enc = build_position_encoding(d_model, 'sine')
        self.pos_scale = build_scale_embedding(4, d_model)
 
    def forward(self, x, mask):

        xs = self.body(x)
        # encode_input, mask
        outs = []
        masks = []

        for name, x in xs.items():
            if name == '0':
                scale_map = self.c3_conv(x)
            elif name == '1':
                scale_map = self.c4_conv(x)
            else:
                scale_map = self.c5_conv(x)

            mask_ = F.interpolate(mask[None].float(), size=scale_map.shape[-2:]).to(torch.bool)[0]
            outs.append(scale_map)
            masks.append(mask_)

        c6 = self.c6_conv(xs['2'])
        mask_ = F.interpolate(mask[None].float(), size=c6.shape[-2:]).to(torch.bool)[0]
        outs.append(c6)
        masks.append(mask_)

        poses = []

        #positions
        for i in range(len(outs)):
            pos_enc_ = self.pos_enc(outs[i], masks[i])
            pos_scale_ =self.pos_scale(outs[i], i)
            pos_ = pos_enc_ + pos_scale_
            poses.append(pos_)

        return outs, masks, poses


    def setup(self, args=None):
        cfg = get_cfg()
        self.add_attribute_config(cfg)
        cfg.MODEL.RESNETS.RES5_DILATION = 1
        cfg.freeze()
        default_setup(cfg, args)
        return cfg

    def add_attribute_config(self,cfg):
        cfg._BASE_= "Base-RCNN-grid.yaml"
        cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395]
        cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/FAIR/X-101-32x8d.pkl"
        cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
        cfg.MODEL.RESNETS.NUM_GROUPS = 32
        cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 8
        cfg.MODEL.RESNETS.DEPTH = 101
        cfg.MODEL.RESNETS.OUT_FEATURES = ['res5']
        cfg.MODEL.ATTRIBUTE_ON = False
        cfg.INPUT.MAX_ATTR_PER_INS = 16
        cfg.MODEL.ROI_ATTRIBUTE_HEAD = CN()
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.OBJ_EMBED_DIM = 256
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.FC_DIM = 512
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.LOSS_WEIGHT = 0.2
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_CLASSES = 400
        cfg.MODEL.RPN.BBOX_LOSS_WEIGHT = 1.0
        cfg.MODEL.ROI_BOX_HEAD.BBOX_LOSS_WEIGHT = 1.0
    
    def load_model_(self):
        self.backbone.load_state_dict(torch.load(self.path_))













