"""Synthetic geometry dataset for testing the pipeline, never a substitute for real training labels."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def generate(root,n=12,size=128,seed=11):
    root=Path(root)
    for p in ['images','instance_masks','splits']:(root/p).mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    train=[];val=[]
    for i in range(n):
        img=np.full((size,size,3),rng.uniform(12,50),dtype=np.float32)
        ids=np.zeros((size,size),dtype=np.uint16)
        blocks=[(10,20,61,49),(61 if i%2==0 else 68,20,119,49),(16,66,102,104)]
        for k,(x1,y1,x2,y2) in enumerate(blocks,1):
            patch=rng.normal(177,22,size=(y2-y1,x2-x1,3)).astype(np.float32)
            img[y1:y2,x1:x2]=patch
            ids[y1:y2,x1:x2]=k
        img=cv2.GaussianBlur(img,(3,3),0.6)
        img=np.clip(img,0,255).astype(np.uint8)
        name=f'{i:04d}'
        cv2.imwrite(str(root/'images'/f'{name}.png'),img)
        cv2.imwrite(str(root/'instance_masks'/f'{name}.png'),ids)
        (val if i>=max(1,n-2) else train).append(name)
    (root/'splits'/'train.txt').write_text('\n'.join(train)+'\n')
    (root/'splits'/'val.txt').write_text('\n'.join(val)+'\n')
    print(f'Synthetic samples: train={len(train)}, val={len(val)} under {root}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='data');p.add_argument('--n',type=int,default=12)
    a=p.parse_args();generate(a.root,a.n)
