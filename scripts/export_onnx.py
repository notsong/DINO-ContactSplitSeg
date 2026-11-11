"""Export coarse model only. Requires onnx; some DINOv3 operators may need compatibility adaptations."""
import argparse
import torch
from contactsplit.model import ContactSplitSeg


class CoarseExport(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model

    def forward(self,images):
        p=self.model.forward_coarse(images)
        return p['pred_logits'],p['pred_masks'],p['boundary_logits'],p['contact_logits'],p['center_logits']


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='coarse_segmenter.onnx')
    p.add_argument('--size',type=int,default=768)
    args=p.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=ContactSplitSeg(state['config'])
    model.load_state_dict(state['model']);model.eval()
    torch.onnx.export(CoarseExport(model).eval(),(torch.randn(1,3,args.size,args.size),),args.output,input_names=['images'],output_names=['class_logits','mask_logits','boundary_logits','contact_logits','center_logits'],opset_version=18,dynamo=False)
    print('Exported:',args.output,'(validate numeric parity before production)')
