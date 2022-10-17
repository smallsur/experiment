import torch
import torch.nn.functional as F
from torch import nn

from utils import box_cxcywh_to_xyxy


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