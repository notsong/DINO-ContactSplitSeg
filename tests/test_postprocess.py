import numpy as np
from contactsplit.postprocess import split_touching,mask_iou


def test_contact_split_requires_evidence():
    mask=np.zeros((80,80),dtype=bool)
    mask[10:60,10:70]=True
    contact=np.zeros((80,80),dtype=np.float32)
    center=np.zeros((80,80),dtype=np.float32)
    center[30,25]=1;center[30,55]=1
    assert len(split_touching(mask,contact,center,min_area=50,min_contact_pixels=6))==1
    contact[12:58,40:42]=1
    chunks=split_touching(mask,contact,center,min_area=50,min_contact_pixels=6)
    assert len(chunks)==2
    assert np.logical_or.reduce(chunks).sum()==mask.sum()
    assert (chunks[0]&chunks[1]).sum()==0


def test_refine_inference_path():
    import torch
    from contactsplit.config import load_config
    from contactsplit.postprocess import decode_predictions
    from contactsplit.refine import ROIRefiner
    cfg=load_config('configs/smoke.yaml')
    cfg['inference']['use_refiner']=True
    cfg['inference']['use_contact_split']=False
    h=w=64
    masks=torch.full((1,8,16,16),-10.)
    masks[:,0,4:12,4:12]=10.
    cls=torch.full((1,8,2),-8.)
    cls[0,0,0]=8.
    pred={'pred_masks':masks,'pred_logits':cls,
          'boundary_logits':torch.zeros(1,1,16,16),
          'contact_logits':torch.zeros(1,1,16,16),
          'center_logits':torch.zeros(1,1,16,16)}
    refiner=ROIRefiner().eval()
    meta={'original_hw':(64,64),'resized_hw':(64,64),'size':64}
    result=decode_predictions(pred,torch.rand(3,64,64),meta,cfg,refiner)
    assert result['instance_map'].shape==(64,64)
