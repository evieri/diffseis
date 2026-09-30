import os
import sys
import torch
from torch.cuda.amp import autocast
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.unet import UNet

device = 'cuda'
model = UNet(in_channel=2, out_channel=1).to(device)

def hook(m, i, o, n=""):
    if torch.isnan(o).any() or torch.isinf(o).any():
        print(f"[{n}] OUTPUT has NaN/Inf! ")
        print("Output:", o.flatten()[:10])
        print("Input:", i[0].flatten()[:10])
        print("Weight:", m.weight.flatten()[:10])
        sys.exit(0)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        module.register_forward_hook(lambda m, i, o, n=name: hook(m, i, o, n))

x_cond = torch.randn(2, 1, 64, 128, device=device) * 100.0
x_noisy = torch.randn(2, 1, 64, 128, device=device) * 100.0
x = torch.cat([x_cond, x_noisy], dim=1)
time = torch.rand(2, device=device)

with autocast(enabled=True):
    out = model(x, time)
