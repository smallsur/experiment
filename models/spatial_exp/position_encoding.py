# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

# from util.misc import NestedTensor
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

# class PositionEmbeddingSine(nn.Module):
#     """
#     This is a more standard version of the position embedding, very similar to the one
#     used by the Attention is all you need paper, generalized to work on images.
#     """
#     def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
#         super().__init__()
#         self.num_pos_feats = num_pos_feats
#         self.temperature = temperature
#         self.normalize = normalize
#         if scale is not None and normalize is False:
#             raise ValueError("normalize should be True if scale is passed")
#         if scale is None:
#             scale = 2 * math.pi
#         self.scale = scale

#     def forward(self, x, mask):
#         assert mask is not None
#         bs ,l = x.shape[:2]
#         mask = mask.squeeze(1).squeeze(1).view(bs,7,7)
#         not_mask = ~mask
#         y_embed = not_mask.cumsum(1, dtype=torch.float32)
#         x_embed = not_mask.cumsum(2, dtype=torch.float32)
#         if self.normalize:
#             eps = 1e-6
#             y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
#             x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

#         dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
#         dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

#         pos_x = x_embed[:, :, :, None] / dim_t
#         pos_y = y_embed[:, :, :, None] / dim_t
#         pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
#         pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
#         return pos.view(bs, self.num_pos_feats*2, -1).permute(0, 2, 1)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):

        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        return pos


class ScaleEmbeddingLearned(nn.Module):

    def __init__(self, scales=4, dim=256):
        super().__init__()
        self.scale_embed = nn.Embedding(scales, embedding_dim=dim)

    def reset_parameters(self):
        nn.init.uniform_(self.scale_embed.weight)

    def forward(self, tensor_list, scale: int):
        #B, C, H, W = tensor_list
        scale_embed = self.scale_embed(torch.tensor([scale], dtype=torch.long,
                                                    device=tensor_list.device))
        # 1, C, 1, 1
        scale_embed = scale_embed.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return scale_embed
    

def build_position_encoding(hidden_dim=512,position_embedding='sine'):

    N_steps = hidden_dim // 2 
    if position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        position_embedding = PositionEmbeddingLearned(N_steps)

    return position_embedding


def build_scale_embedding(scales = 4, d_model=512):

    scale_embedding = ScaleEmbeddingLearned(scales=scales, dim=d_model)
    
    return scale_embedding