from pathlib import Path
import cv2
import numpy as np
import torch
from .model import ContactSplitSeg
from .dataset import prepare_image
from .postprocess import decode_predictions


class ContactSplitPredictor:
    def __init__(self,checkpoint,device=None):
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.cfg=state['config']
        self.model=ContactSplitSeg(self.cfg).to(self.device)
        self.model.load_state_dict(state['model'],strict=True)
        self.model.eval()
        self.stage=state.get('stage',0)

    @torch.inference_mode()
    def predict_rgb(self,image):
        t,meta=prepare_image(image,self.cfg['train']['image_size'])
        t=t.to(self.device)
        pred=self.model(t.unsqueeze(0))
        refiner=self.model.refine_head if self.stage>=3 else None
        return decode_predictions(pred,t,meta,self.cfg,refiner)

    def predict_file(self,path):
        image=cv2.imread(str(path))
        if image is None:raise FileNotFoundError(path)
        return self.predict_rgb(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))


def save_result(result,out):
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    labels=result['instance_map']
    if labels.max()>=65535:raise ValueError('More than 65534 instances: use .npy instead')
    cv2.imwrite(str(folder/'instance_map.png'),labels.astype(np.uint16))
    import json
    (folder/'instances.json').write_text(json.dumps(result['instances'],ensure_ascii=False),encoding='utf-8')
    from .visualize import render_instance_overlay
    render_instance_overlay(result,folder/'instance_overlay.png')
