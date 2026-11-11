"""Export the refinement model independently for batch ROI serving."""
import argparse
import torch
from contactsplit.model import ContactSplitSeg

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='roi_refiner.onnx')
    p.add_argument('--size',type=int,default=256)
    a=p.parse_args()
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if int(state.get('stage',0))<3:
        raise RuntimeError('Checkpoint has no trained ROI refiner (requires stage >=3)')
    model=ContactSplitSeg(state['config']).eval()
    model.load_state_dict(state['model'])
    if model.refine_head is None:raise ValueError('Refinement disabled')
    torch.onnx.export(model.refine_head,torch.rand(1,6,a.size,a.size),a.output,
                      input_names=['roi_rgb_mask_boundary_contact'],output_names=['refined_mask_logits'],
                      dynamic_axes={'roi_rgb_mask_boundary_contact':{0:'num_rois'},'refined_mask_logits':{0:'num_rois'}},
                      opset_version=18,dynamo=False)
    print(f'Exported {a.output}')
