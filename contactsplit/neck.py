"""ViT features -> high-resolution FPN. P2 fusion also sees the original pixels."""
import torch
from torch import nn
import torch.nn.functional as F


def conv(in_ch, out_ch, stride=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False), nn.GroupNorm(min(16, out_ch), out_ch), nn.GELU())


class MultiScaleNeck(nn.Module):
    def __init__(self, in_channels=384, channels=192):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(in_channels, channels, 1) for _ in range(4)])
        self.shallow = nn.Sequential(conv(3, channels//2, 2), conv(channels//2, channels, 2))
        self.lateral = nn.ModuleList([conv(channels, channels) for _ in range(4)])
        self.refine = nn.ModuleList([conv(channels, channels) for _ in range(4)])

    def forward(self, images, feats):
        if len(feats) != 4:
            raise ValueError('Neck requires 4 feature maps')
        h, w = images.shape[-2:]
        sizes = [(h//4,w//4),(h//8,w//8),(h//16,w//16),(h//32,w//32)]
        p2,p3,p4,p5 = [self.lateral[i](F.interpolate(self.proj[i](feat),size=sizes[i],mode='bilinear',align_corners=False)) for i,feat in enumerate(feats)]
        p5 = self.refine[3](p5)
        p4 = self.refine[2](p4 + F.interpolate(p5,size=p4.shape[-2:],mode='nearest'))
        p3 = self.refine[1](p3 + F.interpolate(p4,size=p3.shape[-2:],mode='nearest'))
        p2 = self.refine[0](p2 + F.interpolate(p3,size=p2.shape[-2:],mode='nearest') + self.shallow(images))
        return {'p2':p2,'p3':p3,'p4':p4,'p5':p5}
