"""Generate instance, outer-boundary, true-contact, and one-center-per-instance targets."""
import cv2
import numpy as np
import torch


def _dilate(binary, radius=1):
    width = 2*radius+1
    return cv2.dilate(binary.astype(np.uint8),np.ones((width,width),np.uint8))>0


def make_targets(instance_map, ignore_id=65535, contact_width=3):
    ids = np.asarray(instance_map)
    if ids.ndim != 2 or ids.dtype.kind not in 'iu':
        raise ValueError('instance_map must be an integer HxW array')
    h,w = ids.shape
    valid = ids != ignore_id
    labels = [int(k) for k in np.unique(ids) if k != 0 and k != ignore_id]
    masks = np.stack([ids==k for k in labels]) if labels else np.zeros((0,h,w),dtype=bool)
    binary = ((ids>0)&valid).astype(np.uint8)
    # Only the foreground/background interface; true instance-contact edges are separate.
    eroded = cv2.erode(binary,np.ones((3,3),np.uint8))
    boundary = (binary-eroded).astype(bool)
    contact = np.zeros((h,w),dtype=bool)
    diff_h = (ids[:,1:]!=ids[:,:-1])&(ids[:,1:]>0)&(ids[:,:-1]>0)&valid[:,1:]&valid[:,:-1]
    contact[:,1:] |= diff_h
    contact[:,:-1] |= diff_h
    diff_v = (ids[1:,:]!=ids[:-1,:])&(ids[1:,:]>0)&(ids[:-1,:]>0)&valid[1:,:]&valid[:-1,:]
    contact[1:,:] |= diff_v
    contact[:-1,:] |= diff_v
    if contact_width>1:
        contact = _dilate(contact,max(1,contact_width//2))
    center = np.zeros((h,w),dtype=np.float32)
    centers = []
    boxes = []
    for mask in masks:
        ys,xs = np.where(mask)
        boxes.append([int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)])
        dist=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
        cy,cx = np.unravel_index(int(dist.argmax()),dist.shape)
        centers.append((int(cx),int(cy)))
        sigma = max(2., min(10., float(dist[cy,cx])*0.35))
        rad = int(np.ceil(3*sigma))
        y1,y2=max(0,cy-rad),min(h,cy+rad+1)
        x1,x2=max(0,cx-rad),min(w,cx+rad+1)
        yy,xx=np.ogrid[y1:y2,x1:x2]
        g=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma*sigma))
        np.maximum(center[y1:y2,x1:x2],g,out=center[y1:y2,x1:x2])
    return {
        'instance_map':torch.from_numpy(ids.astype(np.int64)),
        'valid_mask':torch.from_numpy(valid),
        'masks':torch.from_numpy(masks.astype(bool)),
        'labels':torch.zeros((len(labels),),dtype=torch.long),
        'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
        'boundary':torch.from_numpy(boundary.astype('float32'))[None],
        'contact':torch.from_numpy(contact.astype('float32'))[None],
        'center':torch.from_numpy(center)[None],
        'centers':centers,
    }
