"""DINOv3 adapter. Weights must be obtained separately from the official source."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class MockBackbone(nn.Module):
    """Small conv backbone for CPU tests only; NOT equivalent to DINOv3."""
    out_channels = 64

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(),
        )
        self.layers = nn.ModuleList([nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.GELU()) for _ in range(4)])

    def forward(self, x):
        x = self.stem(x)
        result = []
        for layer in self.layers:
            x = layer(x)
            result.append(x)
        return result


class DINOv3Backbone(nn.Module):
    out_channels = 384

    def __init__(self, repo, weights):
        super().__init__()
        if not Path(repo).is_dir():
            raise FileNotFoundError(f'DINOv3 source missing: {repo}. Clone facebookresearch/dinov3 locally.')
        if not Path(weights).is_file():
            raise FileNotFoundError(f'DINOv3 weights missing: {weights}. Request official weights first.')
        self.model = torch.hub.load(str(Path(repo).resolve()), 'dinov3_vits16', source='local', weights=str(Path(weights).resolve()))

    def forward(self, x):
        feats = self.model.get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=True, norm=True)
        if len(feats) != 4:
            raise RuntimeError('Expected exactly 4 DINOv3 feature maps')
        return list(feats)


def make_backbone(cfg):
    name = cfg['model'].get('backbone', 'dinov3')
    if name == 'mock':
        return MockBackbone()
    if name == 'dinov3':
        return DINOv3Backbone(cfg['model']['dinov3_repo'], cfg['model']['dinov3_weights'])
    raise ValueError(f'Unknown backbone: {name}')
