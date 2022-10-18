import torch
import os
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.backbone import build_backbone
from torchvision.models._utils import IntermediateLayerGetter
from .position_encoding import build_position_encoding, build_scale_embedding

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.config import CfgNode as CN

class Backbone(nn.Module):

    def __init__(self, args, train_backbone=False,  d_model = 256, last_dim=2048, return_interm_layers=True):

        super(Backbone, self).__init__()

        self.backbone = build_backbone(self.setup(args)) # 72*96, 36*48, 18*24, 8*11

        self.path_ = os.path.join(args.dir_to_save_model, 'rs101.pth')


        for name, parameter in self.backbone.named_parameters():
            if not train_backbone or 'res3' not in name and 'res4' not in name and 'res5' not in name:
                parameter.requires_grad_(False)

        if return_interm_layers:
            self.return_layers = {"res3": "0", "res4": "1", "res5": "2"}
        else:
            self.return_layers = {"res5": "0"}

        self.body = IntermediateLayerGetter(self.backbone, return_layers=self.return_layers)

        self.pos_enc = build_position_encoding(d_model, 'sine')

        if return_interm_layers:
            self.c3_conv = nn.Conv2d(last_dim//4, d_model, kernel_size=1)
            self.c4_conv = nn.Conv2d(last_dim//2, d_model, kernel_size=1)
            self.c5_conv = nn.Conv2d(last_dim, d_model, kernel_size=1)
            self.c6_conv = nn.Conv2d(last_dim, d_model, kernel_size=(3,3), stride=2)
            self.pos_scale = build_scale_embedding(4, d_model)


        self.return_interm_layers = return_interm_layers

    def forword_deformable_detr(self, x, mask):
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

    def forward_detr(self, x, mask):

        xs = self.body(x)
        outs = []
        poses = []
        masks = []
        for name, x in xs.items():
            mask_ = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            pos = self.pos_enc(x, mask_)
            poses.append(pos)
            outs.append(x)
            masks.append(mask_)
        return outs, masks, poses
        
    def forward(self, x, mask):
        if self.return_interm_layers:
            return self.forword_deformable_detr(x, mask)
        else:
            return self.forward_detr(x, mask)


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
