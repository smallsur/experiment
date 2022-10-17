import torch
import os
import torch.nn.functional as F
from torch import nn
from typing import List, Optional


from models.build import BuildModel


from .detr import Detr_Transformer
from models.captioning_model import CaptioningModel
from .backbone import Backbone


from .evalue_box import evalue_box

def col_box_5(args):

    box_backbone = Detr_Transformer(d_model=256, h=8, num_enc=6, num_dec=6, d_ff=2048, dropout=0.1,
                     num_classes=1601, aux_outputs=args.aux_outputs, num_queries=100, norm =True)

    backbone = Backbone(args, train_backbone=args.train_backbone, d_model=256, last_dim=2048, return_interm_layers=False)

    model = Transformer(bos_idx=2, backbone=backbone, box_backbone=box_backbone, )

    return model

BuildModel.add(7, col_box_5)

class Transformer(CaptioningModel):
    def __init__(self, bos_idx, box_backbone, backbone):
        super(Transformer, self).__init__()

        self.bos_idx = bos_idx

        self.backbone = backbone
        self.box_backbone = box_backbone
 
        self.init_weights()

        self.backbone.load_model_()


    @property
    def d_model(self):
        return self.box_backbone.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, masks, only_box = False):

        features, masks, poses = self.backbone.forward(features, masks)

        out, _, _ = self.box_backbone.forward(features, masks, poses)

        return out

    def forward_box_loss(self, output, target):

        return self.box_backbone.forward_box_loss(output, target)

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]
    
    def dump_dt(self,output,ids,sizes):
        self.box_backbone.dump_dt(output,ids,sizes)

    def evalue_box_(self,args):
        return evalue_box(args)
    
    def dump_gt(self,target,boxfeild):

        return self.box_backbone.dump_gt(target,boxfeild)


