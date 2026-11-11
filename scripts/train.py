import argparse
from contactsplit.config import load_config
from contactsplit.training import train

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/full.yaml')
    p.add_argument('--resume',default=None)
    args=p.parse_args()
    train(load_config(args.config),resume=args.resume)
