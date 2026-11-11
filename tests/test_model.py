import torch
from contactsplit.config import load_config
from contactsplit.model import ContactSplitSeg
from contactsplit.losses import MultiTaskCriterion
from contactsplit.targets import make_targets


def test_forward_and_backward():
    torch.set_num_threads(2)
    cfg=load_config('configs/smoke.yaml')
    model=ContactSplitSeg(cfg)
    inp=torch.rand(1,3,64,64)
    p=model(inp)
    assert p['pred_logits'].shape==(1,8,2)
    assert p['pred_masks'].shape==(1,8,16,16)
    assert p['contact_logits'].shape==(1,1,16,16)
    assert model.forward_refine(torch.rand(1,6,32,32)).shape==(1,1,32,32)
    import numpy as np
    m=np.zeros((64,64),dtype=np.uint16)
    m[4:30,4:31]=1;m[4:30,31:57]=2
    t=make_targets(m)
    losses,_=MultiTaskCriterion(stage=2)(p,[t])
    assert all(torch.isfinite(x) for x in losses.values())
    sum(losses.values()).backward()
    assert any(param.grad is not None for param in model.instance_head.parameters())
