import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.diffusion import GaussianDiffusion, Trainer
from src.unet import UNet
import torch

mode = "demultiple" #demultiple, interpolation, denoising
folder = None # Uses on-the-fly synthetic data
image_size = (64,128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
        in_channel=2,
        out_channel=1
)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)

diffusion = GaussianDiffusion(
    model,
    mode = mode,
    channels = 1,
    image_size = image_size,
    timesteps = 2000,
    loss_type = 'l2', # L1 or L2
).to(device)

trainer = Trainer(
    diffusion,
    mode = mode,
    folder = folder,
    image_size = image_size,
    train_batch_size = 64,
    train_lr = 2e-5,
    train_num_steps = 2000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                      # Hardware without Tensor Cores physical chip (TU117)
)

trainer.train()
trainer.save("final")