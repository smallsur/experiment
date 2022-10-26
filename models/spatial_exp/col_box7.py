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

    box_backbone = Detr_Transformer(d_model=512, h=8, num_enc=6, num_dec=6, d_ff=2048, dropout=0.1,
                     num_classes=1601, aux_outputs=args.aux_outputs, num_queries=100, norm = False, 
                     sequential_input=args.add_features)

    projectlayer = ProjectLayer()

    encoder = Encoder(3, 0)

    # decoder = Decoder(10201, 54, 3, 1, incor= 'single')
    decoder = TransformerDecoderLayer(10201, 54, 3, 1,language_model_path=args.language_model_path)

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
    
    def forward(self, features, masks, shape):
        bs = features.shape[0]
        last_width, last_height = shape
        pos_grid = self.pos_enc(features, masks.squeeze(1).squeeze(1).view(bs, last_width, last_height)).permute(0, 2, 3, 1).contiguous().view(bs, -1, self.d_model)

        out = F.relu(self.fc(features))
        out = self.drop(out)
        out = self.layer_norm(out)

        return out, pos_grid


class Transformer(CaptioningModel):
    def __init__(self, bos_idx, box_backbone, encoder, decoder, project):
        super(Transformer, self).__init__()

        self.bos_idx = bos_idx

        self.box_backbone = box_backbone
        self.encoder = encoder
        self.decoder = decoder

        self.sequential_input = self.box_backbone.sequential_input

        self.project = project

        self.register_state('enc_out', None)
        self.register_state('box_lastlayer', None)
        self.register_state('mask_enc_cap', None)

        self.init_weights()


    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @property
    def d_model(self):
        return self.decoder.d_model

    def forward(self, features, seq, mask_enc_box, shape, only_box = False):
        features_box, features_cap =None, None

        if isinstance(features,torch.Tensor):
            features_box = features
            features_cap = features
        else:
            features_box, features_cap = features

        mask_enc_cap = (torch.sum(features_cap, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)

        features_cap, pos_grid_cap = self.project(features_cap, mask_enc_cap, (7, 7))
        features_box, pos_grid_box = self.project(features_box, mask_enc_box, shape=shape)

        if self.sequential_input:
            f_, m_ = features_cap, mask_enc_cap
        else:
            f_, m_ = features_box, mask_enc_box

        out, box_lastlayer, pos_emb = self.box_backbone.forward(f_, m_, pos_grid_box)

        if only_box:
            return out
        
        enc_out = self.encoder(input=features_cap, attention_mask=mask_enc_cap, pos=pos_grid_cap,  box_output=box_lastlayer, pos_emb=pos_emb)
        dec_out = self.decoder(input=seq, encoder_output=enc_out, box_lastlayer=box_lastlayer, mask_encoder=mask_enc_cap)
        
        return out, dec_out


    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                shape = kwargs['shape']
                features_box, features_cap =None, None
                if not isinstance(visual,list):
                    features_box = visual
                    features_cap = visual
                else:
                    features_box, features_cap = visual
                mask_enc_box = kwargs['mask_enc_box']
                self.mask_enc_cap = (torch.sum(features_cap, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)
                features_cap, pos_grid_cap = self.project(features_cap, self.mask_enc_cap)
                features_box, pos_grid_box = self.project(features_box, mask_enc_box)

                if self.sequential_input:
                    f_, m_ = features_cap, self.mask_enc_cap
                    shape = (7, 7)
                else:
                    f_, m_ = features_box, mask_enc_box

                _, self.box_lastlayer, pos_emb = self.box_backbone.forward(f_, m_, pos_grid_box)
                self.enc_out = self.encoder(input=features_cap, attention_mask=self.mask_enc_cap, pos=pos_grid_cap,  box_output=self.box_lastlayer, pos_emb=pos_emb)

                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, encoder_output=self.enc_out, box_lastlayer=self.box_lastlayer, mask_encoder=self.mask_enc_cap)


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