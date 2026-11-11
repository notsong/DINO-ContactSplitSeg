"""Staged training, validation, and safe checkpointing."""
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .model import ContactSplitSeg
from .dataset import SpecimenDataset,collate_samples,MEAN,STD
from .losses import MultiTaskCriterion,match_queries
from .refine import make_refine_input,crop_roi


def configure_trainable(model,stage,freeze_backbone=False):
    for p in model.parameters():p.requires_grad=True
    if stage<=2 and model.refine_head is not None:
        for p in model.refine_head.parameters():p.requires_grad=False
    if stage==3:
        for p in model.parameters():p.requires_grad=False
        if model.refine_head is None:
            raise ValueError('stage 3 requires enabled refinement head')
        for p in model.refine_head.parameters():p.requires_grad=True
    elif freeze_backbone:
        for p in model.backbone.parameters():p.requires_grad=False


def build_optimizer(model,cfg):
    tr=cfg['train']
    groups=[]
    normal=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')]
    backbone=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')]
    if normal:groups.append({'params':normal,'lr':tr['lr']})
    if backbone:groups.append({'params':backbone,'lr':tr['backbone_lr']})
    if not groups:raise ValueError('No trainable parameters')
    return torch.optim.AdamW(groups,weight_decay=tr.get('weight_decay',0.05))


def refine_training_loss(model,images,targets,outputs,config):
    """Crop GT ROI and predicted coarse masks using GT<->Query matching.
    The coarse path is intentionally frozen in stage 3.
    """
    assigns=match_queries(outputs,targets)
    h,w=images.shape[-2:]
    dense=F.interpolate(torch.cat([outputs['boundary_logits'],outputs['contact_logits']],dim=1),size=(h,w),mode='bilinear',align_corners=False).sigmoid()
    coarse=F.interpolate(outputs['pred_masks'],size=(h,w),mode='bilinear',align_corners=False).sigmoid()
    mean=images.new_tensor(MEAN)[None,:,None,None]
    std=images.new_tensor(STD)[None,:,None,None]
    rgb=(images*std+mean).clamp(0,1)
    inputs=[]
    truths=[]
    roi_size=config['model'].get('roi_size',256)
    for b,(rows,cols) in enumerate(assigns):
        for qi,ti in zip(rows.tolist(),cols.tolist()):
            x1,y1,x2,y2=targets[b]['boxes'][ti].tolist()
            pad=max(6,round(max(x2-x1,y2-y1)*0.1))
            box=(max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad))
            inputs.append(make_refine_input(rgb[b],coarse[b,qi:qi+1],dense[b,0:1],dense[b,1:2],box,roi_size))
            gt=targets[b]['masks'][ti:ti+1].float().to(images.device)
            truths.append(crop_roi(gt,box,roi_size))
    if not inputs:
        return outputs['pred_masks'].sum()*0
    # Limit per-step ROI memory to avoid OOM on dense samples.
    loss=images.new_tensor(0.)
    for start in range(0,len(inputs),8):
        x=torch.stack(inputs[start:start+8])
        y=torch.stack(truths[start:start+8])
        pred=model.forward_refine(x)
        bce=F.binary_cross_entropy_with_logits(pred,y)
        pr=pred.sigmoid()
        dice=1-(2*(pr*y).flatten(1).sum(-1)+1)/(pr.flatten(1).sum(-1)+y.flatten(1).sum(-1)+1)
        loss=loss+(bce+dice.mean())*len(x)
    return loss/len(inputs)


def epoch(model,loader,device,cfg,stage,optimizer=None,scaler=None):
    training=optimizer is not None
    model.train(training)
    if stage==3: model.backbone.eval();model.neck.eval();model.instance_head.eval()
    criterion=MultiTaskCriterion(stage=stage)
    totals=[]
    for images,targets in loader:
        images=images.to(device)
        if training:optimizer.zero_grad(set_to_none=True)
        amp=bool(cfg['train'].get('amp',True)) and device.type=='cuda'
        with torch.set_grad_enabled(training),torch.autocast(device_type=device.type,enabled=amp):
            if stage==3:
                with torch.no_grad():outputs=model(images)
                loss=refine_training_loss(model,images,targets,outputs,cfg)
            else:
                outputs=model(images)
                losses,_=criterion(outputs,targets)
                loss=sum(losses.values())
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                scaler.step(optimizer);scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                optimizer.step()
        totals.append(float(loss.detach().item()))
    return float(np.mean(totals)) if totals else float('nan')


def train(cfg,resume=None):
    tr=cfg['train']
    seed=int(tr.get('seed',42));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ContactSplitSeg(cfg).to(device)
    if resume:
        ckpt=torch.load(resume,map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['model'])
    ds_args={'root':cfg['data']['root'],'size':tr['image_size'],'ignore_id':cfg['data'].get('ignore_id',65535),'contact_width':cfg['data'].get('contact_width',3)}
    train_ds=SpecimenDataset(split='train',augment=True,**ds_args)
    val_ds=SpecimenDataset(split='val',augment=False,**ds_args)
    train_dl=DataLoader(train_ds,batch_size=tr['batch_size'],shuffle=True,num_workers=tr.get('workers',0),collate_fn=collate_samples)
    val_dl=DataLoader(val_ds,batch_size=tr['batch_size'],shuffle=False,num_workers=tr.get('workers',0),collate_fn=collate_samples)
    stage=int(tr.get('stage',2))
    out=Path(tr.get('checkpoint_dir','checkpoints'));out.mkdir(parents=True,exist_ok=True)
    best=float('inf')
    for ep in range(tr['epochs']):
        configure_trainable(model,stage,freeze_backbone=(ep<tr.get('freeze_backbone_epochs',0)))
        optimizer=build_optimizer(model,cfg)
        scaler=torch.amp.GradScaler('cuda',enabled=tr.get('amp',True) and device.type=='cuda')
        loss_train=epoch(model,train_dl,device,cfg,stage,optimizer,scaler)
        loss_val=epoch(model,val_dl,device,cfg,stage)
        print(f'epoch {ep+1}/{tr["epochs"]} train={loss_train:.4f} val={loss_val:.4f}',flush=True)
        state={'model':model.state_dict(),'config':cfg,'stage':stage,'epoch':ep+1,'val_loss':loss_val}
        torch.save(state,out/'last.pt')
        if loss_val<best:
            best=loss_val;torch.save(state,out/'best.pt')
    return model
