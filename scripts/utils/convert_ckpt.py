import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch

def convert_ckpt():
    path = '/mnt/direct-attached/as-cells/rfdetr-ema-09-val_map_ema0.7293.ckpt'
    out_path = '/mnt/direct-attached/as-cells/rfdetr-ema-09-val_map_ema0.7293-converted.pt'
    
    print(f"Loading {path}...")
    ckpt = torch.load(path, map_location='cpu')
    
    new_ckpt = {
        'model': ckpt.get('ema_state_dict', ckpt.get('state_dict'))
    }
    
    # Check if the keys have 'model.model.' prefix which is common in lightning, if so remove 'model.model.' or 'model.'
    # wait, earlier we saw they don't have 'model.' prefix, they start with 'transformer.' or 'class_embed.' directly.
    # Actually wait! The rfdetr package expects full names like 'model.backbone...' or 'model.transformer...'?
    # Let's check a working checkpoint.
    
    torch.save(new_ckpt, out_path)
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    convert_ckpt()