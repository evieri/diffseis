from diffusion import GaussianDiffusion, Trainer
from unet import UNet

mode = "demultiple" #demultiple, interpolation, denoising
folder = "data/"
image_size = (64,128)

model = UNet(
        in_channel=2,
        out_channel=1
).cuda()

diffusion = GaussianDiffusion(
    model,
    mode = mode,
    channels = 1,
    image_size = image_size,
    timesteps = 2000,
    loss_type = 'l1', # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    mode = mode,
    folder = folder,
    image_size = image_size,
    train_batch_size = 8,
    train_lr = 2e-5,
    train_num_steps = 500,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
)

trainer.train()