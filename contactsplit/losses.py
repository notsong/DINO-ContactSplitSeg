"""Hungarian query/GT matching and multi-task criterion."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _resize(x, hw, mode='bilinear'):
    if x.shape[-2:]==hw:
        return x
    return F.interpolate(x,size=hw,mode=mode,align_corners=False if mode=='bilinear' else None)


def _dice_loss(logits, target, valid=None):
    prob=logits.sigmoid()
    if valid is not None:
        prob=prob*valid
        target=target*valid
    numerator=2*(prob*target).flatten(1).sum(-1)+1.
    denominator=prob.flatten(1).sum(-1)+target.flatten(1).sum(-1)+1.
    return (1.-numerator/denominator).mean()


@torch.no_grad()
def match_queries(pred, targets, matching_size=64):
    """One-to-one matching, computed on compact masks to bound memory use."""
    b,q,_,_=pred['pred_masks'].shape
    assignments=[]
    for i in range(b):
        target=targets[i]
        n=target['masks'].shape[0]
        if n>q:
            raise ValueError(f'Target has {n} instances but model has only {q} queries. Increase num_queries or crop images.')
        if n==0:
            assignments.append((torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long)))
            continue
        mask=F.interpolate(pred['pred_masks'][i:i+1],size=(matching_size,matching_size),mode='bilinear',align_corners=False).sigmoid().squeeze(0)
        gt=F.interpolate(target['masks'].float().to(mask.device).unsqueeze(0),size=(matching_size,matching_size),mode='nearest').squeeze(0)
        valid=F.interpolate(target['valid_mask'].float().to(mask.device)[None,None],size=(matching_size,matching_size),mode='nearest').squeeze()
        mask=mask.flatten(1)*valid.flatten()
        gt=gt.flatten(1)*valid.flatten()
        inter=mask @ gt.T
        dice=1.-(2*inter+1)/(mask.sum(-1)[:,None]+gt.sum(-1)[None,:]+1)
        cls_cost = -pred['pred_logits'][i].softmax(-1)[:,0,None]
        cost=(2*dice+cls_cost).detach().cpu().numpy()
        rows,cols=linear_sum_assignment(cost)
        assignments.append((torch.as_tensor(rows,dtype=torch.long),torch.as_tensor(cols,dtype=torch.long)))
    return assignments


def _instance_loss(pred,targets,assignments):
    dev=pred['pred_masks'].device
    b,q,h,w=pred['pred_masks'].shape
    logits=pred['pred_logits']
    ce_target=torch.ones((b,q),device=dev,dtype=torch.long)
    losses=[]
    for i,(rows,cols) in enumerate(assignments):
        rows=rows.to(dev)
        cols=cols.to(dev)
        if rows.numel()==0:
            continue
        ce_target[i,rows]=0
        gt=_resize(targets[i]['masks'][cols].float().to(dev).unsqueeze(0),(h,w),mode='nearest').squeeze(0)
        valid=_resize(targets[i]['valid_mask'].float().to(dev)[None,None],(h,w),mode='nearest').squeeze(0)
        pm=pred['pred_masks'][i,rows]
        bce=F.binary_cross_entropy_with_logits(pm,gt,reduction='none')
        losses.append((bce*valid).sum()/(valid.sum().clamp_min(1)*rows.numel()) + _dice_loss(pm,gt,valid))
    ce=F.cross_entropy(logits.transpose(1,2),ce_target,weight=torch.tensor([1.,0.1],device=dev))
    seg=torch.stack(losses).mean() if losses else logits.sum()*0
    return ce+seg


def _dense_loss(logits,targets,key,is_center=False):
    device=logits.device
    gt=torch.stack([t[key] for t in targets]).to(device)
    valid=torch.stack([t['valid_mask'] for t in targets]).to(device)[:,None].float()
    gt=_resize(gt,logits.shape[-2:],'nearest')
    valid=_resize(valid,logits.shape[-2:],'nearest')
    if is_center:
        return (((logits.sigmoid()-gt)**2)*(1+9*gt)*valid).sum()/((1+9*gt).mul(valid).sum().clamp_min(1))
    # In all-negative samples, BCE provides valid supervision and Dice is omitted.
    pos=(gt*valid).sum()
    neg=((1-gt)*valid).sum()
    weight=torch.where(gt>0.5,torch.clamp(neg/(pos+1),min=1,max=20),torch.ones_like(gt))
    bce=(F.binary_cross_entropy_with_logits(logits,gt,reduction='none')*weight*valid).sum()/(weight*valid).sum().clamp_min(1)
    dice=_dice_loss(logits,gt,valid) if pos.item()>0 else logits.sum()*0
    return bce+dice


class MultiTaskCriterion:
    def __init__(self,stage=2):
        self.stage=stage

    def __call__(self,pred,targets):
        assignments=match_queries(pred,targets)
        losses={'instance':_instance_loss(pred,targets,assignments)}
        if pred.get('aux_outputs'):
            # Fixed matched indices as in common DETR auxiliary supervision.
            losses['instance_aux']=sum(_instance_loss(p,targets,assignments) for p in pred['aux_outputs'])/len(pred['aux_outputs'])*0.5
        if self.stage>=2:
            losses['boundary']=_dense_loss(pred['boundary_logits'],targets,'boundary')*0.5
            losses['contact']=_dense_loss(pred['contact_logits'],targets,'contact')
            losses['center']=_dense_loss(pred['center_logits'],targets,'center',True)*0.5
        return losses, assignments
