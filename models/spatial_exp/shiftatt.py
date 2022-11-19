import torch
import os
import torch.nn.functional as F
from torch import nn
from typing import List, Optional
from models.transformer.utils import restore_scale

class ShiftHeadAttention(nn.Module):
    def __init__(self, h, d_model, k, scales=1, dropout=0.1):
        super(ShiftHeadAttention, self).__init__()
        assert h == 8 
        assert d_model % h == 0
        
        self.d_k = int(d_model / h)
        self.h = h

        self.q_proj = nn.Linear(d_model, d_model)
        self.b_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        self.dropout = None

        if self.dropout:
            self.dropout = nn.Dropout(p=dropout)

        self.k = k
        self.scales = scales

        self.offset_dims = 2 * self.h * self.k * self.scales
        self.A_dims = self.h * self.k * self.scales

        # 2MLK for offsets MLK for A_mlqk
        self.offset_proj = nn.Linear(d_model, self.offset_dims)
        self.A_proj = nn.Linear(d_model, self.A_dims)

        self.wm_proj = nn.Linear(d_model, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.offset_proj.weight, 0.0)
        torch.nn.init.constant_(self.A_proj.weight, 0.0)

        torch.nn.init.constant_(self.A_proj.bias, 1 / (self.scales * self.k))

        torch.nn.init.xavier_uniform(self.q_proj.weight)
        torch.nn.init.xavier_uniform(self.b_proj.weight)
        torch.nn.init.xavier_uniform(self.k_proj.weight)
        torch.nn.init.xavier_uniform(self.wm_proj.weight)
        
        def init_xy(bias, x, y):
            torch.nn.init.constant_(bias[:, 0], float(x))
            torch.nn.init.constant_(bias[:, 1], float(y))

        # caution: offset layout will be  M, L, K, 2
        bias = self.offset_proj.bias.view(self.h, self.scales, self.k, 2)

        init_xy(bias[0], x=-self.k, y=-self.k)
        init_xy(bias[1], x=-self.k, y=0)
        init_xy(bias[2], x=-self.k, y=self.k)
        init_xy(bias[3], x=0, y=-self.k)
        init_xy(bias[4], x=0, y=self.k)
        init_xy(bias[5], x=self.k, y=-self.k)
        init_xy(bias[6], x=self.k, y=0)
        init_xy(bias[7], x=self.k, y=self.k)

    def forward(self,
                query: torch.Tensor,
                keys: List[torch.Tensor],
                ref_point: torch.Tensor,
                query_mask: torch.Tensor = None,
                key_masks: Optional[torch.Tensor] = None,
                src_query = None):

        if key_masks is None:
            key_masks = [None] * len(keys)

        assert len(keys) == self.scales


        nbatches, query_height, query_width, _ = query.shape

        # B, H, W, C
        query = self.q_proj(query)
        #
        src_query = self.b_proj(src_query)

        # B, H, W, 2MLK
        offset = self.offset_proj(src_query)
        # B, H, W, M, 2LK
        offset = offset.view(nbatches, query_height, query_width, self.h, -1)

        # B, H, W, MLK
        A = self.A_proj(src_query)

        # B, H, W, 1, mask before softmax
        if query_mask is not None:
            query_mask_ = query_mask.unsqueeze(dim=-1)
            _, _, _, mlk = A.shape
            query_mask_ = query_mask_.expand(nbatches, query_height, query_width, mlk)
            A = torch.masked_fill(A, mask=query_mask_, value=float('-inf'))

        # B, H, W, M, LK
        A = A.view(nbatches, query_height, query_width, self.h, -1)
        A = F.softmax(A, dim=-1)

        # mask nan position
        if query_mask is not None:
            # B, H, W, 1, 1
            query_mask_ = query_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            A = torch.masked_fill(A, query_mask_.expand_as(A), 0.0)


        offset = offset.view(nbatches, query_height, query_width, self.h, self.scales, self.k, 2)
        offset = offset.permute(0, 3, 4, 5, 1, 2, 6).contiguous()
        # B*M, L, K, H, W, 2
        offset = offset.view(nbatches * self.h, self.scales, self.k, query_height, query_width, 2)

        A = A.permute(0, 3, 1, 2, 4).contiguous()
        # B*M, H*W, LK
        A = A.view(nbatches * self.h, query_height * query_width, -1)

        scale_features = []
        for l in range(self.scales):
            feat_map = keys[l]
            _, h, w, _ = feat_map.shape

            key_mask = key_masks[l]

            # B, H, W, 2
            reversed_ref_point = restore_scale(height=h, width=w, ref_point=ref_point)

            # B, H, W, 2 -> B*M, H, W, 2
            reversed_ref_point = reversed_ref_point.repeat(self.h, 1, 1, 1)

            # B, h, w, M, C_v
            scale_feature = self.k_proj(feat_map).view(nbatches, h, w, self.h, self.d_k)

            if key_mask is not None:
                # B, h, w, 1, 1
                key_mask = key_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
                key_mask = key_mask.expand(nbatches, h, w, self.h, self.d_k)
                scale_feature = torch.masked_fill(scale_feature, mask=key_mask, value=0)

            # B, M, C_v, h, w
            scale_feature = scale_feature.permute(0, 3, 4, 1, 2).contiguous()
            # B*M, C_v, h, w
            scale_feature = scale_feature.view(-1, self.d_k, h, w)

            k_features = []

            for k in range(self.k):
                points = reversed_ref_point + offset[:, l, k, :, :, :]
                vgrid_x = 2.0 * points[:, :, :, 0] / max(w - 1, 1) - 1.0
                vgrid_y = 2.0 * points[:, :, :, 1] / max(h - 1, 1) - 1.0
                vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

                # B*M, C_v, H, W
                feat = F.grid_sample(scale_feature, vgrid_scaled, mode='bilinear', padding_mode='zeros')
                k_features.append(feat)
            # B*M, k, C_v, H, W
            k_features = torch.stack(k_features, dim=1)
            scale_features.append(k_features)

        # B*M, L, K, C_v, H, W
        scale_features = torch.stack(scale_features, dim=1)

        # B*M, H*W, C_v, LK
        scale_features = scale_features.permute(0, 4, 5, 3, 1, 2).contiguous()
        scale_features = scale_features.view(nbatches * self.h, query_height * query_width, self.d_k, -1)

        # B*M, H*W, C_v
        feat = torch.einsum('nlds, nls -> nld', scale_features, A)

        # B*M, H*W, C_v -> B, M, H, W, C_v
        feat = feat.view(nbatches, self.h, query_height, query_width, self.d_k)
        # B, M, H, W, C_v -> B, H, W, M, C_v
        feat = feat.permute(0, 2, 3, 1, 4).contiguous()
        # B, H, W, M, C_v -> B, H, W, M * C_v
        feat = feat.view(nbatches, query_height, query_width, self.d_k * self.h)

        feat = self.wm_proj(feat)
        if self.dropout:

            feat = self.dropout(feat)

        return feat