from .attention import MultiHeadAttention,ScaledDotProductAttention
from .encoders import EncoderLayer
from .decoders import DecoderLayer
from .transformer import Transformer
from models.transformer.utils import sinusoid_encoding_table,PositionWiseFeedForward
from models.build import BuildModel
from torch import nn

import torch
from models.containers import ModuleList,Module
from torch.nn import functional as F



from models.captioning_model import CaptioningModel

def build_rl_two(args):
    return Transformer_rl()

BuildModel.add(2,build_rl_two)

class Mid_layer_rl(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1):
        super(Mid_layer_rl, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
    
    def forward(self,enc_output, dec_output):
        out = self.self_att(enc_output,dec_output,dec_output)
        out = self.pwff(out)

        return out


class Transformer_rl(CaptioningModel):
    def __init__(self, bos_idx=2, N=3,d_in=2048,d_model=512,d_k=64,
                    padding_idx_de = 1,vocab_size=10201,max_len=54,padding_idx_en=0,
                    d_v=64,h=8,d_ff=2048, dropout=.1,identity_map_reordering=False):
        super(Transformer_rl, self).__init__()
        self.bos_idx = bos_idx
        self.dmodel = d_model
        self.dropout = dropout
        #encoder
        self.layers_en = ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering)
                                     for _ in range(N)])
        self.fc_en = nn.Linear(d_in, d_model)
        self.dropout_en = nn.Dropout(p=dropout)
        self.layer_norm_en = nn.LayerNorm(d_model)

        #decoder
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx_de)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)

        self.layers_de = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N)])
        self.fc_de = nn.Linear(d_model, vocab_size, bias=False)

        self.mid_layers = ModuleList([Mid_layer_rl(d_model, d_k, d_v, h, d_ff, dropout) for _ in range(N-1)])

        self.max_len = max_len
        self.padding_idx_de = padding_idx_de
        self.padding_idx_en = padding_idx_en
        self.N = N

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

        # self.register_state('enc_output', None)
        # self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.dmodel

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encoder_step(self,images):
        enc_output = F.relu(self.fc_en(images))
        enc_output = self.dropout_en(enc_output)
        enc_output = self.layer_norm_en(enc_output)

        attention_mask = (torch.sum(images, -1) == self.padding_idx_en).unsqueeze(1).unsqueeze(1)

        return enc_output,attention_mask
    
    def decoder_step(self,input):
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx_de).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx_de).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)

        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq
            
        dec_output = self.word_emb(input) + self.pos_emb(seq)

        return dec_output,mask_queries,mask_self_attention

    def forward(self, images, input):
        # enc_output, mask_enc = self.encoder(images)
        # dec_output = self.decoder(seq, enc_output, mask_enc)
        enc_output = F.relu(self.fc_en(images))
        enc_output = self.dropout_en(enc_output)
        enc_output = self.layer_norm_en(enc_output)

        attention_mask = (torch.sum(images, -1) == self.padding_idx_en).unsqueeze(1).unsqueeze(1)

        # enc_output,attention_mask = self.encoder_step(images)

        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx_de).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx_de).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)

        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq
            
        dec_output = self.word_emb(input) + self.pos_emb(seq)

        # dec_output,mask_queries,mask_self_attention = self.decoder_step(input)
        enc_output = self.layers_en[0](enc_output,enc_output,enc_output,attention_mask)
        enc_output = self.layers_en[1](enc_output,enc_output,enc_output,attention_mask)
        dec_output = self.layers_de[0](dec_output,enc_output,mask_queries,mask_self_attention,attention_mask)
        enc_output = self.mid_layers[0](enc_output,dec_output)
        enc_output = self.layers_en[2](enc_output,enc_output,enc_output,attention_mask)
        dec_output = self.layers_de[1](dec_output,enc_output,mask_queries,mask_self_attention,attention_mask)
        enc_output = self.mid_layers[1](enc_output,dec_output)
        dec_output = self.layers_de[1](dec_output,enc_output,mask_queries,mask_self_attention,attention_mask)

        # for i in range(self.N-1):
        #     enc_output = self.layers_en[i](enc_output,enc_output,enc_output,attention_mask)
        #     dec_output = self.layers_de[i](dec_output,enc_output,mask_queries,mask_self_attention,attention_mask)
        #     enc_output = self.mid_layers[i](enc_output,dec_output)

        # enc_output = self.layers_en[self.N-1](enc_output,enc_output,enc_output,attention_mask)
        # dec_output = self.layers_de[self.N-1](dec_output,enc_output,mask_queries,mask_self_attention,attention_mask)

        out = self.fc_de(dec_output)

        return F.softmax(out,-1)

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        # if mode == 'teacher_forcing':
        #     raise NotImplementedError
        # elif mode == 'feedback':
            # if t == 0:
            #     self.enc_output, self.mask_enc = self.encoder(visual)
            #     if isinstance(visual, torch.Tensor):
            #         it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
            #     else:
            #         it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            # else:
            #     it = prev_output
        if t == 0:
            if isinstance(visual, torch.Tensor):
                it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
            else:
                it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            
        else:
            it = prev_output
        

        # enc_output,attention_mask = self.encoder_step(visual)
        # dec_output,mask_queries,mask_self_attention = self.decoder_step(it)
        # for i in range(self.N-1):
        #     enc_output = self.layers_en[i](enc_output,enc_output,enc_output,attention_mask)
        #     dec_output = self.layers_de[i](dec_output,enc_output,mask_queries,mask_self_attention,attention_mask)
        #     enc_output = self.mid_layers[i](enc_output,dec_output)

        # enc_output = self.layers_en[self.N-1](enc_output,enc_output,enc_output,attention_mask)
        # dec_output = self.layers_de[self.N-1](dec_output,enc_output,mask_queries,mask_self_attention,attention_mask)
        # out = self.fc_de(dec_output)
        # enc_output, mask_enc = self.encoder(visual)
        # dec_output = self.decoder(seq, enc_output, mask_enc)
        enc_output = F.relu(self.fc_en(visual))
        enc_output = self.dropout_en(enc_output)
        enc_output = self.layer_norm_en(enc_output)

        attention_mask = (torch.sum(visual, -1) == self.padding_idx_en).unsqueeze(1).unsqueeze(1)

        # enc_output,attention_mask = self.encoder_step(images)
        input = it
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx_de).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx_de).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)

        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq
            
        dec_output = self.word_emb(input) + self.pos_emb(seq)

        # dec_output,mask_queries,mask_self_attention = self.decoder_step(input)

        # for i in range(self.N-1):
        #     enc_output = self.layers_en[i](enc_output,enc_output,enc_output,attention_mask)
        #     dec_output = self.layers_de[i](dec_output,enc_output,mask_queries,mask_self_attention,attention_mask)
        #     enc_output = self.mid_layers[i](enc_output,dec_output)

        # enc_output = self.layers_en[self.N-1](enc_output,enc_output,enc_output,attention_mask)
        # dec_output = self.layers_de[self.N-1](dec_output,enc_output,mask_queries,mask_self_attention,attention_mask)

        enc_output = self.layers_en[0](enc_output,enc_output,enc_output,attention_mask)
        enc_output = self.layers_en[1](enc_output,enc_output,enc_output,attention_mask)
        dec_output = self.layers_de[0](dec_output,enc_output,mask_queries,mask_self_attention,attention_mask)
        enc_output = self.mid_layers[0](enc_output,dec_output)
        enc_output = self.layers_en[2](enc_output,enc_output,enc_output,attention_mask)
        dec_output = self.layers_de[1](dec_output,enc_output,mask_queries,mask_self_attention,attention_mask)
        enc_output = self.mid_layers[1](enc_output,dec_output)
        dec_output = self.layers_de[1](dec_output,enc_output,mask_queries,mask_self_attention,attention_mask)
        out = self.fc_de(dec_output)

        return F.softmax(out,-1)

