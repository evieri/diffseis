from diffusion import GaussianDiffusion, Trainer
from unet import UNet
import torch

mode = "demultiple" #demultiple, interpolation, denoising
folder = "data/"
image_size = (64,128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
        in_channel=2,
        out_channel=1
).to(device)

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
    train_batch_size = 4,
    train_lr = 2e-5,
    train_num_steps = 2000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = torch.cuda.is_available(),  # turn on mixed precision only if CUDA is available
)

trainer.train()
trainer.save("final")