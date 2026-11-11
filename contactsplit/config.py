from pathlib import Path
import yaml


def load_config(path):
    with Path(path).open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or 'model' not in cfg:
        raise ValueError('Missing model configuration')
    return cfg
