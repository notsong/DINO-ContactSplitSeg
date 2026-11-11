import torch
from torch import nn
from .backbone import make_backbone
from .neck import MultiScaleNeck
from .heads import MaskQueryHead,DenseHead
from .refine import ROIRefiner


class ContactSplitSeg(nn.Module):
    def __init__(self,config):
        super().__init__()
        m=config['model']
        channels=m.get('hidden_dim',192)
        self.backbone=make_backbone(config)
        self.neck=MultiScaleNeck(self.backbone.out_channels,channels)
        self.instance_head=MaskQueryHead(channels,m.get('num_queries',64),m.get('decoder_layers',6),m.get('heads',8))
        self.boundary_head=DenseHead(channels)
        self.contact_head=DenseHead(channels)
        self.center_head=DenseHead(channels)
        self.refine_head=ROIRefiner() if m.get('use_refiner',True) else None

    def forward_coarse(self,images):
        pyramid=self.neck(images,self.backbone(images))
        pred=self.instance_head(pyramid)
        pred['boundary_logits']=self.boundary_head(pyramid['p2'])
        pred['contact_logits']=self.contact_head(pyramid['p2'])
        pred['center_logits']=self.center_head(pyramid['p2'])
        return pred

    def forward_refine(self,rois):
        if self.refine_head is None:
            raise RuntimeError('Refinement module disabled')
        return self.refine_head(rois)

    def forward(self,images):
        return self.forward_coarse(images)
