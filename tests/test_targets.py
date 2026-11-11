import numpy as np
from contactsplit.targets import make_targets


def test_contact_vs_gap():
    m=np.zeros((32,32),dtype=np.uint16)
    m[8:20,3:12]=1
    m[8:20,12:23]=2
    out=make_targets(m,contact_width=1)
    assert out['masks'].shape==(2,32,32)
    assert out['contact'].sum()>0
    assert out['center'].max() == 1.
    assert out['boundary'].sum()>0
    m[:,12:14]=0
    out=make_targets(m,contact_width=1)
    assert out['contact'].sum()==0


def test_ignore_and_empty():
    m=np.zeros((16,16),dtype=np.uint16)
    m[4:8,4:8]=65535
    out=make_targets(m)
    assert out['masks'].shape==(0,16,16)
    assert not out['valid_mask'][5,5]
    assert out['boxes'].shape==(0,4)
