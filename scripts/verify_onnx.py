"""PyTorch / ONNX Runtime numerical parity for a trained ROI refiner."""
import argparse
import numpy as np
import torch
from contactsplit.model import ContactSplitSeg

if __name__=='__main__':
    import onnxruntime as ort
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--onnx',required=True)
    p.add_argument('--size',type=int,default=256)
    a=p.parse_args()
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model=ContactSplitSeg(ckpt['config']).eval()
    model.load_state_dict(ckpt['model'])
    rng=np.random.default_rng(123)
    input_array=rng.standard_normal((2,6,a.size,a.size)).astype('float32')
    with torch.inference_mode():
        pytorch=model.refine_head(torch.from_numpy(input_array)).numpy()
    session=ort.InferenceSession(a.onnx,providers=['CPUExecutionProvider'])
    output=session.run(None,{session.get_inputs()[0].name:input_array})[0]
    diff=np.abs(pytorch-output)
    print(f'max_abs={diff.max():.8f}, mean_abs={diff.mean():.8f}')
    np.testing.assert_allclose(pytorch,output,atol=1e-4,rtol=1e-3)
