import cv2
import numpy as np


def render_instance_overlay(result,destination,background=None):
    labels=result['instance_map']
    h,w=labels.shape
    rng=np.random.default_rng(14)
    colors=np.zeros((int(labels.max())+1,3),dtype=np.uint8)
    if len(colors)>1:colors[1:]=rng.integers(65,256,size=(len(colors)-1,3),dtype=np.uint8)
    canvas=colors[labels]
    if background is not None:
        if background.shape[:2]!=(h,w):raise ValueError('Overlay shape mismatch')
        canvas=(canvas*0.65+background.astype(np.float32)*0.35).astype(np.uint8)
    for item in result['instances']:
        contours,_=cv2.findContours((labels==item['id']).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas,contours,-1,(255,255,255),1)
    cv2.imwrite(str(destination),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
    return canvas
