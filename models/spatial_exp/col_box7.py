from turtle import width
import torch
import os
import torch.nn.functional as F
from torch import nn

from models.build import BuildModel

from .shfitcaption import Encoder, Decoder
from .detr_local import Detr_Transformer
from models.captioning_model import CaptioningModel
from .position_encoding import build_position_encoding
from models.rstnet.decoders import TransformerDecoderLayer

from .evalue_box import evalue_box

def col_box_7(args):

    box_backbone = Detr_Transformer(d_model=512, h=8, num_enc=3, num_dec=3, d_ff=2048, dropout=0.1,
                     num_classes=1601, aux_outputs=args.aux_outputs, num_queries=100, norm = False, 
                     sequential_input=~args.add_features)

    projectlayer = ProjectLayer()

    encoder = Encoder(3, 0)

    decoder = Decoder(10201, 54, 3, 1, incor= 'single')
    # decoder = TransformerDecoderLayer(10201, 54, 3, 1,language_model_path=args.language_model_path)

    model = Transformer(bos_idx=2,  box_backbone=box_backbone, encoder=encoder, decoder=decoder, project=projectlayer)

    return model

BuildModel.add(9, col_box_7)


class ProjectLayer(nn.Module):
    def __init__(self, d_in=2048, d_model=512, dropout=0.1):
        super(ProjectLayer, self).__init__()
        self.fc = nn.Linear(d_in, d_model)
        self.drop = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.pos_enc = build_position_encoding(512, 'sine')

        self.d_model =d_model
        self.init_weights()


    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, features, shape):
        bs = features.shape[0]
        mask = (torch.sum(features, -1) == 0)
        last_width, last_height = shape
        pos_grid = self.pos_enc(features, mask.view(bs, last_width, last_height)).permute(0, 2, 3, 1).contiguous().view(bs, -1, self.d_model)

        out = F.relu(self.fc(features))
        out = self.drop(out)
        out = self.layer_norm(out)

        mask = mask.unsqueeze(1).unsqueeze(1)

        return out, pos_grid, mask


class Transformer(CaptioningModel):
    def __init__(self, bos_idx, box_backbone, encoder, decoder, project):
        super(Transformer, self).__init__()

        self.bos_idx = bos_idx

        self.box_backbone = box_backbone
        self.encoder = encoder
        self.decoder = decoder

        self.project = project

        self.register_state('enc_out', None)
        self.register_state('pos_grid', None)
        self.register_state('mask_enc', None)
        self.register_state('box_last', None)

    @property
    def d_model(self):
        return self.decoder.d_model

    def forward(self, images, seq, only_box = False):
        features, pos_grid, mask_enc = self.project(images, (7,7))

        box_out, box_last, pos_emb = self.box_backbone(features, mask_enc, pos_grid)

        if only_box:
            return box_out
        enc_out = self.encoder(features, mask_enc, pos_grid, attention_weights=None, box_output=box_last, pos_emb = pos_emb)
        dec_out = self.decoder(seq, enc_out, box_last, mask_enc, pos_grid =pos_grid)

        return box_out, dec_out


    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                features, self.pos_grid, self.mask_enc = self.project(visual, (7,7))

                _, self.box_last, pos_emb = self.box_backbone(features, self.mask_enc, self.pos_grid)

                self.enc_out = self.encoder(features, self.mask_enc, self.pos_grid, attention_weights=None, box_output=self.box_last, pos_emb = pos_emb)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_out, self.box_last, self.mask_enc, self.pos_grid)


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