"""Mask2Former-style masked cross-attention decoder plus dense auxiliary heads.

Not the upstream Facebook implementation; intentionally small and self-contained.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .neck import conv


class DecoderLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.linear1 = nn.Linear(dim, dim*4)
        self.linear2 = nn.Linear(dim*4, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, queries, memory, attn_mask=None):
        out = self.cross(queries, memory, memory, attn_mask=attn_mask, need_weights=False)[0]
        queries = self.norm1(queries+out)
        queries = self.norm2(queries+self.self_attn(queries,queries,queries,need_weights=False)[0])
        return self.norm3(queries+self.linear2(F.gelu(self.linear1(queries))))


class MaskQueryHead(nn.Module):
    def __init__(self, channels=192, queries=64, layers=6, heads=8):
        super().__init__()
        self.queries = nn.Embedding(queries, channels)
        self.layers = nn.ModuleList([DecoderLayer(channels, heads) for _ in range(layers)])
        self.classifier = nn.Linear(channels, 2)
        self.mask_embed = nn.Sequential(nn.Linear(channels,channels),nn.ReLU(),nn.Linear(channels,channels))
        self.pixel_embed = conv(channels,channels)
        self.heads = heads

    def _predict(self, queries, pixels):
        cls = self.classifier(queries)
        embed = self.mask_embed(queries)
        masks = torch.einsum('bqc,bchw->bqhw',embed,pixels) / (pixels.shape[1]**0.5)
        return {'pred_logits':cls,'pred_masks':masks}

    def forward(self, pyramid):
        p2 = self.pixel_embed(pyramid['p2'])
        b = p2.shape[0]
        queries = self.queries.weight[None].expand(b,-1,-1)
        aux = []
        pred = self._predict(queries,p2)
        for i,layer in enumerate(self.layers):
            feat = pyramid[['p5','p4','p3'][i%3]]
            # Bound attention memory for large images (avoid quadratic growth).
            if max(feat.shape[-2:]) > 64:
                feat = F.adaptive_avg_pool2d(feat,(min(64,feat.shape[-2]),min(64,feat.shape[-1])))
            memory = feat.flatten(2).transpose(1,2)
            mask = F.interpolate(pred['pred_masks'].detach(),size=feat.shape[-2:],mode='bilinear',align_corners=False)
            ignore = (mask.sigmoid().flatten(2)<0.5)
            # Never mask all pixels for a query.
            ignore = ignore & (~ignore.all(dim=-1,keepdim=True))
            attn_mask = ignore.repeat_interleave(self.heads,dim=0)
            queries = layer(queries,memory,attn_mask=attn_mask)
            aux.append(pred)
            pred = self._predict(queries,p2)
        pred['aux_outputs'] = aux[1:]  # exclude learned unconditioned query prediction
        return pred


class DenseHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(conv(channels,channels//2),conv(channels//2,channels//2),nn.Conv2d(channels//2,1,1))

    def forward(self, p2):
        return self.net(p2)
