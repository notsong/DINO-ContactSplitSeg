from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .targets import make_targets

MEAN = np.array([0.485,0.456,0.406],dtype=np.float32)
STD = np.array([0.229,0.224,0.225],dtype=np.float32)


def prepare_image(image, size):
    """Letterbox to square; returns normalized RGB Tensor and coordinate metadata."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected RGB HxWx3 image')
    h,w = image.shape[:2]
    scale=min(size/h,size/w)
    rh,rw=max(1,round(h*scale)),max(1,round(w*scale))
    resized=cv2.resize(image,(rw,rh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    canvas=np.zeros((size,size,3),dtype=np.uint8)
    canvas[:rh,:rw] = resized
    arr=(canvas.astype(np.float32)/255.-MEAN)/STD
    return torch.from_numpy(arr.transpose(2,0,1).copy()), {'original_hw':(h,w),'resized_hw':(rh,rw),'scale_xy':(rw/w,rh/h),'size':size}


class SpecimenDataset(Dataset):
    def __init__(self, root, split='train', size=768, augment=False, ignore_id=65535, contact_width=3):
        self.root=Path(root)
        self.size=size
        self.augment=augment
        self.ignore_id=ignore_id
        self.contact_width=contact_width
        path=self.root/'splits'/f'{split}.txt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing split list: {path}')
        self.ids=[line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not self.ids:
            raise ValueError(f'No images listed in {path}')

    def __len__(self):
        return len(self.ids)

    def _resolve(self, sub, name):
        folder=self.root/sub
        for suffix in ['','.png','.jpg','.jpeg','.tif','.tiff']:
            p=folder/(name+suffix)
            if p.exists():return p
        raise FileNotFoundError(f'Cannot find {name} under {folder}')

    def __getitem__(self,i):
        name=self.ids[i]
        img_path=self._resolve('images',name)
        mask_path=self._resolve('instance_masks',name)
        img=cv2.imread(str(img_path),cv2.IMREAD_COLOR)
        ids=cv2.imread(str(mask_path),cv2.IMREAD_UNCHANGED)
        if img is None or ids is None or ids.ndim!=2:
            raise ValueError(f'Invalid image or instance mask for {name}')
        if img.shape[:2]!=ids.shape:
            raise ValueError(f'Image/Mask size mismatch for {name}')
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if self.augment:
            k=random.randint(0,3)
            img=np.rot90(img,k).copy()
            ids=np.rot90(ids,k).copy()
            if random.random()<0.5:
                img=np.fliplr(img).copy()
                ids=np.fliplr(ids).copy()
            if random.random()<0.65:
                img=np.clip(img.astype(np.float32)*random.uniform(0.8,1.2)+random.uniform(-12,12),0,255).astype(np.uint8)
        tensor,meta=prepare_image(img,self.size)
        rh,rw=meta['resized_hw']
        # Padding is ignored, not treated as background training data.
        mask=np.full((self.size,self.size),self.ignore_id,dtype=np.int32)
        mask[:rh,:rw]=cv2.resize(ids.astype(np.int32),(rw,rh),interpolation=cv2.INTER_NEAREST)
        target=make_targets(mask,self.ignore_id,self.contact_width)
        target['image_id']=name
        target['meta']=meta
        return tensor,target


def collate_samples(batch):
    images,targets=zip(*batch)
    return torch.stack(images),list(targets)
