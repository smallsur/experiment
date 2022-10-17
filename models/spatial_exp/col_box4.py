import torch
import os
import torch.nn.functional as F
from torch import nn
from typing import List, Optional


from models.build import BuildModel


from .deformable_detr import Detr_Transformer, Backbone
from models.containers import ModuleList, Module
from models.captioning_model import CaptioningModel

from .evalue_box import evalue_box

def col_box_4(args):

    box_backbone = Detr_Transformer(d_model=256, h=8, num_enc=6, num_dec=6, d_ff=2048, dropout=0.1,
                     scales=4, k=4, last_feat_height=16, last_feat_width=16, 
                    num_classes=1601, aux_outputs=args.aux_outputs, num_queries=100, norm =True)

    backbone = Backbone(args, train_backbone=args.train_backbone, d_model=256, last_dim=2048)

    model = Transformer(bos_idx=2, backbone=backbone, box_backbone=box_backbone, )

    return model


BuildModel.add(6, col_box_4)


class Transformer(CaptioningModel):
    def __init__(self, bos_idx, box_backbone, backbone):
        super(Transformer, self).__init__()

        self.bos_idx = bos_idx

        self.backbone = backbone
        self.box_backbone = box_backbone
 
        self.init_weights()


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


