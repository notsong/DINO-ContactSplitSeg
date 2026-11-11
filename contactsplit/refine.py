"""Instance-conditioned ROI segmentation (6 channels RGB + coarse + boundary + contact)."""
import torch
from torch import nn
import torch.nn.functional as F
from .neck import conv


class ROIRefiner(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.enc1=nn.Sequential(conv(6,channels),conv(channels,channels))
        self.enc2=nn.Sequential(nn.MaxPool2d(2),conv(channels,channels*2))
        self.enc3=nn.Sequential(nn.MaxPool2d(2),conv(channels*2,channels*4))
        self.dec2=conv(channels*6,channels*2)
        self.dec1=conv(channels*3,channels)
        self.out=nn.Conv2d(channels,1,1)

    def forward(self,roi):
        if roi.shape[1]!=6:
            raise ValueError('ROIRefiner expects 6 channels')
        e1=self.enc1(roi)
        e2=self.enc2(e1)
        e3=self.enc3(e2)
        d2=self.dec2(torch.cat([F.interpolate(e3,size=e2.shape[-2:],mode='bilinear',align_corners=False),e2],dim=1))
        d1=self.dec1(torch.cat([F.interpolate(d2,size=e1.shape[-2:],mode='bilinear',align_corners=False),e1],dim=1))
        return self.out(d1)


def crop_roi(tensor, box, size=256):
    """Differentiable crop from [C,H,W], box in pixel xyxy, into [C,S,S]."""
    if tensor.ndim!=3:
        raise ValueError('Expected CHW tensor')
    _,h,w=tensor.shape
    x1,y1,x2,y2=[float(v) for v in box]
    x1,x2=max(0,x1),min(w,x2)
    y1,y2=max(0,y1),min(h,y2)
    if x2<=x1 or y2<=y1:
        raise ValueError('Empty ROI')
    yy,xx=torch.meshgrid(torch.linspace(y1,y2-1,size,device=tensor.device),torch.linspace(x1,x2-1,size,device=tensor.device),indexing='ij')
    grid=torch.stack(((xx+0.5)*2/w-1,(yy+0.5)*2/h-1),dim=-1).unsqueeze(0)
    return F.grid_sample(tensor.unsqueeze(0),grid,mode='bilinear',padding_mode='border',align_corners=False).squeeze(0)


def make_refine_input(image,mask,boundary,contact,box,size):
    """Inputs are CHW, mask/boundary/contact must be [1,H,W]."""
    # All channels use the same grid; returns [6,S,S].
    return crop_roi(torch.cat([image,mask,boundary,contact],dim=0),box,size)
