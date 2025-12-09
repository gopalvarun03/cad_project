import torch
import sys

ckpt_path = "proj_log/pretrained/model/ckpt_epoch1000.pth"
try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print("Checkpoint keys:", ckpt.keys())
    if 'net' in ckpt:
        print("Net keys:", list(ckpt['net'].keys())[:5])
    else:
        # Maybe it's the state dict directly?
        print("First 5 keys:", list(ckpt.keys())[:5])
except Exception as e:
    print(e)
