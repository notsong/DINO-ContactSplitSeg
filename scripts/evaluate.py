"""Evaluate foreground IoU, count error and GT/pred instance IoU matching.
For research-quality AP/PQ, integrate a COCO/Panoptic evaluator separately.
"""
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment
from contactsplit.predictor import ContactSplitPredictor
from contactsplit.dataset import SpecimenDataset


def scores(pred,gt):
    pred_ids=[k for k in np.unique(pred) if k>0]
    gt_ids=[k for k in np.unique(gt) if k>0 and k!=65535]
    fg_valid=gt!=65535
    a=(pred>0)&fg_valid;b=(gt>0)&fg_valid
    foreground_iou=np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum())
    if len(pred_ids) and len(gt_ids):
        iou=np.zeros((len(pred_ids),len(gt_ids)))
        for i,p in enumerate(pred_ids):
            for j,g in enumerate(gt_ids):
                pi=(pred==p)&fg_valid;gi=(gt==g)
                iou[i,j]=np.logical_and(pi,gi).sum()/max(1,np.logical_or(pi,gi).sum())
        rows,cols=linear_sum_assignment(-iou)
        mean_instance_iou=float(np.mean(iou[rows,cols]))
        matched50=int(sum(iou[rows,cols]>=0.5))
    else:
        mean_instance_iou=0.;matched50=0
    return {'foreground_iou':float(foreground_iou),'mean_matched_iou':mean_instance_iou,'matched_50':matched50,'count_error':abs(len(pred_ids)-len(gt_ids))}

if __name__=='__main__':
    import cv2
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--split',default='val')
    a=p.parse_args()
    model=ContactSplitPredictor(a.checkpoint)
    dataset=SpecimenDataset(a.root,a.split,model.cfg['train']['image_size'])
    allscores=[]
    for name in dataset.ids:
        img=dataset._resolve('images',name);gt=dataset._resolve('instance_masks',name)
        pred=model.predict_file(img)['instance_map']
        true=cv2.imread(str(gt),cv2.IMREAD_UNCHANGED)
        allscores.append(scores(pred,true))
    print({k:round(float(np.mean([s[k] for s in allscores])),4) for k in allscores[0]})
