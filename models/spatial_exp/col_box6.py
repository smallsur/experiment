import torch
import os
import torch.nn.functional as F
from torch import nn

from models.build import BuildModel

from .detr_local import Detr_Transformer
from models.captioning_model import CaptioningModel

from .evalue_box import evalue_box


def col_box_6(args):

    box_backbone = Detr_Transformer(d_model=256, h=8, num_enc=6, num_dec=6, d_ff=2048, dropout=0.1,
                     num_classes=1601, aux_outputs=args.aux_outputs, num_queries=100, norm = False, 
                     sequential_input=args.add_features)

    model = Transformer(bos_idx=2,  box_backbone=box_backbone, )

    return model

BuildModel.add(8, col_box_6)



class Transformer(CaptioningModel):
    def __init__(self, bos_idx, box_backbone):
        super(Transformer, self).__init__()

        self.bos_idx = bos_idx

        self.box_backbone = box_backbone
 
        self.init_weights()


    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @property
    def d_model(self):
        return self.box_backbone.d_model

    def forward(self, features, masks, only_box = False):

        out, _, _ = self.box_backbone.forward(features, masks)

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
