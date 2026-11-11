import cv2
import numpy as np
from contactsplit.dataset import SpecimenDataset,collate_samples,prepare_image


def test_dataset_roundtrip(tmp_path):
    for sub in ['images','instance_masks','splits']:(tmp_path/sub).mkdir()
    m=np.zeros((37,53),dtype=np.uint16);m[4:26,5:28]=1
    cv2.imwrite(str(tmp_path/'images'/'a.png'),np.full((37,53,3),160,dtype=np.uint8))
    cv2.imwrite(str(tmp_path/'instance_masks'/'a.png'),m)
    (tmp_path/'splits'/'train.txt').write_text('a\n')
    ds=SpecimenDataset(tmp_path,'train',size=64)
    img,t=ds[0]
    assert img.shape==(3,64,64)
    assert t['masks'].shape[0]==1
    assert t['instance_map'][t['valid_mask']].max()==1
    assert collate_samples([ds[0]])[0].shape==(1,3,64,64)
