import argparse
from collections import Counter
import numpy as np
from contactsplit.dataset import SpecimenDataset

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--split',default='train')
    p.add_argument('--size',type=int,default=768)
    a=p.parse_args()
    ds=SpecimenDataset(a.root,a.split,a.size)
    counts=[];touching=0
    for idx in range(len(ds)):
        _,target=ds[idx]
        counts.append(int(target['masks'].shape[0]))
        touching+=bool(target['contact'].sum()>0)
    print(f'images={len(ds)}, instances min/median/max={min(counts)}/{np.median(counts)}/{max(counts)}, images with contact={touching}')
