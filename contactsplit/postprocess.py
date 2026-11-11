"""Convert query masks into disjoint instance IDs; conservative seed-guided splitting."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import watershed
from .dataset import MEAN,STD
from .refine import make_refine_input


def mask_iou(a,b):
    intersect=np.logical_and(a,b).sum()
    return float(intersect)/max(1,np.logical_or(a,b).sum())


def find_peaks(center,min_score=0.25, min_distance=12):
    """Greedy NMS on local heatmap peaks, sorted by probability."""
    hm=center.astype(np.float32)
    local=(hm==cv2.dilate(hm,np.ones((5,5),np.uint8)))&(hm>=min_score)
    y,x=np.nonzero(local)
    order=np.argsort(-hm[y,x])
    keep=[]
    for k in order:
        p=(int(x[k]),int(y[k]))
        if all((p[0]-q[0])**2+(p[1]-q[1])**2>=min_distance**2 for q in keep):
            keep.append(p)
    return keep


def split_touching(mask,contact,center,min_area=150,min_contact_pixels=12,min_center_score=0.25):
    """Split only with >=2 distinct center peaks AND internal contact evidence.
    If evidence is insufficient, return unchanged mask. Never introduce a background seam.
    """
    mask=np.asarray(mask,dtype=bool)
    if mask.sum()<2*min_area:
        return [mask]
    peaks=[p for p in find_peaks(center,min_center_score,max(8,round(np.sqrt(mask.sum())*0.22))) if mask[p[1],p[0]]]
    if len(peaks)<2:
        return [mask]
    # Require convincing separation in the interior (not just boundary contact elsewhere).
    interior=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
    if np.logical_and(contact>0.55,interior).sum()<min_contact_pixels:
        return [mask]
    markers=np.zeros(mask.shape,dtype=np.int32)
    for i,(x,y) in enumerate(peaks[:6],1):markers[y,x]=i
    dist=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    dnorm=dist/(dist.max()+1e-6)
    elevation=0.7*contact.astype(np.float32)+0.3*(1-dnorm)
    labels=watershed(elevation,markers,mask=mask)
    regions=[labels==i for i in range(1,len(peaks[:6])+1)]
    regions=[r for r in regions if r.sum()>=min_area]
    # Reject if any fragment is too small or a foreground pixel is lost.
    if len(regions)<2 or np.logical_or.reduce(regions).sum()!=mask.sum():
        return [mask]
    return regions


def _up(x,size):
    return F.interpolate(x,size=size,mode='bilinear',align_corners=False)


@torch.no_grad()
def decode_predictions(pred, input_tensor, meta, cfg, refiner=None):
    """Accepts a SINGLE image model output, returns original-resolution ID map + metadata."""
    inf=cfg['inference']
    h,w=meta['resized_hw']
    oh,ow=meta['original_hw']
    score=pred['pred_logits'][0].softmax(-1)[:,0]
    masks=_up(pred['pred_masks'][:1],(meta['size'],meta['size']))[0].sigmoid()[:,:h,:w]
    boundary=_up(pred['boundary_logits'],(meta['size'],meta['size']))[0,0].sigmoid()[:h,:w]
    contact=_up(pred['contact_logits'],(meta['size'],meta['size']))[0,0].sigmoid()[:h,:w]
    center=_up(pred['center_logits'],(meta['size'],meta['size']))[0,0].sigmoid()[:h,:w]
    candidates=[]
    for i in torch.argsort(score,descending=True).tolist():
        s=score[i]
        if s.item()<inf.get('score_threshold',0.45):continue
        prob=masks[i].cpu().numpy()
        binary=prob>=inf.get('mask_threshold',0.5)
        if binary.sum()<inf.get('min_area',120):continue
        if any(mask_iou(binary,c['binary'])>inf.get('nms_iou',0.75) for c in candidates):continue
        candidates.append({'query':i,'score':s.item(),'binary':binary,'prob':prob})
    candidates.sort(key=lambda c:-c['score'])
    bmap=boundary.cpu().numpy()
    cmap=contact.cpu().numpy()
    center_map=center.cpu().numpy()
    # Independent per-instance refinement, using model-space image + dense maps.
    if refiner is not None and inf.get('use_refiner',True):
        unorm = input_tensor[:,:h,:w]*torch.as_tensor(STD,device=input_tensor.device)[:,None,None]+torch.as_tensor(MEAN,device=input_tensor.device)[:,None,None]
        for c in candidates:
            ys,xs=np.where(c['binary'])
            x1,x2=max(0,int(xs.min())-10),min(w,int(xs.max())+11)
            y1,y2=max(0,int(ys.min())-10),min(h,int(ys.max())+11)
            roi=make_refine_input(unorm,torch.from_numpy(c['prob'])[None].to(unorm.device),boundary.unsqueeze(0),contact.unsqueeze(0),(x1,y1,x2,y2),cfg['model'].get('roi_size',256))
            refined=refiner(roi.unsqueeze(0)).sigmoid()
            up=F.interpolate(refined,(y2-y1,x2-x1),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            c['prob'][y1:y2,x1:x2]=up
            c['binary']=c['prob']>=inf.get('mask_threshold',0.5)
    # Pixelwise highest-scoring assignment enforces mutually exclusive instances.
    labels=np.zeros((h,w),dtype=np.int32)
    confidence=np.zeros((h,w),dtype=np.float32)
    for i,c in enumerate(candidates,1):
        weight=c['prob']*c['score']
        update=c['binary']&(weight>confidence)
        labels[update]=i
        confidence[update]=weight[update]
    if inf.get('use_contact_split',True):
        nextid=len(candidates)+1
        for instance_id in range(1,len(candidates)+1):
            m=labels==instance_id
            parts=split_touching(m,cmap,center_map,inf.get('split_min_area',150),inf.get('split_min_contact_pixels',12),inf.get('split_min_center_score',0.25))
            if len(parts)>1:
                for part in parts[1:]:
                    labels[part]=nextid
                    nextid+=1
    labels=cv2.resize(labels.astype(np.int32),(ow,oh),interpolation=cv2.INTER_NEAREST)
    instances=[]
    for k in np.unique(labels):
        if k==0:continue
        binary=(labels==k).astype(np.uint8)
        ys,xs=np.where(binary)
        contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        instances.append({'id':int(k),'bbox':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],'area_pixels':int(binary.sum()),'contours':[c[:,0,:].tolist() for c in contours], 'score':float(candidates[k-1]['score']) if k<=len(candidates) else None, 'split_source':'query' if k<=len(candidates) else 'contact_split'})
    return {'instance_map':labels,'instances':instances,'auxiliary':{'boundary_map':cv2.resize(bmap,(ow,oh)),'contact_map':cv2.resize(cmap,(ow,oh)),'center_map':cv2.resize(center_map,(ow,oh))}}
