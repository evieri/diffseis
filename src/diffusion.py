import math
import copy
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import random

from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
import os

from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
import obspy

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        mode,
        channels,
        image_size,
        timesteps = 2000,
        loss_type='l1',
    ):
        super().__init__()
        self.mode = mode
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.timesteps = timesteps
        
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

        to_torch = partial(torch.tensor, dtype=torch.float32)

        betas = make_beta_schedule(schedule='linear', n_timestep=timesteps, linear_start=1e-6, linear_end=1e-2)
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance( x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn(x.size(), device='cpu').to(x.device) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()
    
    @torch.no_grad()
    def p_sample_loop(self, x_in, mask=None):
        device = self.betas.device
        x_cond = x_in
        if mask is not None:  
            x_cond = x_in*mask
            
        shape = x_cond.shape
        img = torch.randn(shape, device='cpu').to(device)
        ret_img = x_cond
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Inferência no Dado Real', total=self.num_timesteps):
            img = self.p_sample(img, i, condition_x=x_cond)
            if mask is not None:
                img = x_cond + img*(1.-mask)
            
        if mask is not None:
            ret_img = torch.cat([ret_img, x_in], dim=0)
        ret_img = torch.cat([ret_img, img], dim=0)
        return ret_img
    
    @torch.no_grad()
    def inference(self, x_in, mask=None):
        return self.p_sample_loop(x_in, mask)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn(x_start.size(), device='cpu').to(x_start.device))

        # random gama
        return (continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise)

    def p_losses(self, x_cond, x_start, noise=None):
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t-1],self.sqrt_alphas_cumprod_prev[t],size=b)).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn(x_start.size(), device='cpu').to(x_start.device))

        x_noisy = self.q_sample(x_start=x_start,continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if self.mode == "interpolation":
            # here x_cond -> mask
            x_recon = self.denoise_fn(torch.cat([x_start*x_cond, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
            loss = F.mse_loss(noise, x_recon)
        else:
            x_recon = self.denoise_fn(torch.cat([x_cond, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
            loss_mse = F.mse_loss(noise, x_recon)
            
            # Recuprar a estimativa do sinal limpo (x_0)
            sqrt_alpha = continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_sq = (1 - continuous_sqrt_alpha_cumprod**2).sqrt().view(-1, 1, 1, 1)
            x_start_pred = (x_noisy - sqrt_one_minus_alpha_sq * x_recon) / sqrt_alpha
            
            # O ruído geofísico predito (múltiplas)
            predicted_multiple = x_cond - x_start_pred
            
            # Clamp para evitar overflow de inf ao elevar ao quadrado no float32
            x_start_pred = torch.clamp(x_start_pred, min=-50.0, max=50.0)
            predicted_multiple = torch.clamp(predicted_multiple, min=-50.0, max=50.0)
            
            # Flatten para Pearson (força float32 para evitar underflow no AMP)
            x_s_flat = x_start_pred.view(b, -1).float()
            p_m_flat = predicted_multiple.view(b, -1).float()
            x_start_f = x_start.float()
            x_start_pred_f = x_start_pred.float()
            
            mean_s = x_s_flat.mean(dim=1, keepdim=True)
            mean_m = p_m_flat.mean(dim=1, keepdim=True)
            
            x_s_centered = x_s_flat - mean_s
            p_m_centered = p_m_flat - mean_m
            
            cov = (x_s_centered * p_m_centered).sum(dim=1)
            eps = 1e-8
            std_s = torch.sqrt((x_s_centered**2).sum(dim=1) + eps)
            std_m = torch.sqrt((p_m_centered**2).sum(dim=1) + eps)
            
            corr = cov / (std_s * std_m + eps)
            loss_corr = torch.abs(corr).mean()
            
            # RMS Penalty
            rms_pred = torch.sqrt((x_start_pred_f**2).mean(dim=[1,2,3]) + eps)
            rms_true = torch.sqrt((x_start_f**2).mean(dim=[1,2,3]) + eps)
            loss_rms = torch.abs(rms_pred - rms_true).mean()
            
            lambda_corr = 0.1
            lambda_rms = 0.05
            loss = loss_mse.float() + lambda_corr * loss_corr + lambda_rms * loss_rms
            
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
    

# dataset classes


import h5py
from typing import Tuple

def ricker_wavelet(f0: float = 25.0, dt: float = 0.004, length: int = 11) -> np.ndarray:
    t = np.arange(-(length // 2), length // 2 + 1) * dt
    y = (1.0 - 2.0 * (np.pi ** 2) * (f0 ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (f0 ** 2) * (t ** 2))
    return y.astype(np.float32)

def generate_seismic_gather(gather_size=(256, 256), patch_size=(128, 128)):
    ntraces, nsamples = gather_size
    reflections = np.zeros(gather_size, dtype=np.float32)
    diffractions = np.zeros(gather_size, dtype=np.float32)
    
    num_events = np.random.randint(3, 8)
    horizons = []
    
    for _ in range(num_events):
        start_time = np.random.randint(30, nsamples - 30)
        amplitude = np.random.uniform(0.7, 1.0)
        f0_random = np.random.uniform(10.0, 60.0)
        wavelet = ricker_wavelet(f0=f0_random)
        half_w = len(wavelet) // 2
        
        freq_sin = np.random.uniform(0.5, 3.0) / ntraces
        phase_sin = np.random.uniform(0, 2*np.pi)
        amp_sin = np.random.uniform(5.0, 25.0)
        slope = np.random.uniform(-0.2, 0.2)
        
        horizon_t = np.zeros(ntraces, dtype=int)
        for tr in range(ntraces):
            t_exact = start_time + slope * tr + amp_sin * np.sin(2 * np.pi * freq_sin * tr + phase_sin)
            t_idx = int(np.round(t_exact))
            t_idx = max(0, min(nsamples - 1, t_idx))
            horizon_t[tr] = t_idx
            
            amp_variation = np.random.uniform(0.8, 1.2)
            trace_amp = amplitude * amp_variation
            
            if half_w <= t_idx < nsamples - half_w:
                reflections[tr, t_idx - half_w : t_idx + half_w + 1] += wavelet * trace_amp
        horizons.append(horizon_t)
        
    num_diffractions = np.random.randint(5, 15)
    for _ in range(num_diffractions):
        if len(horizons) > 0:
            hor_idx = np.random.randint(0, len(horizons))
            apex_x = np.random.randint(10, ntraces - 10)
            t0 = horizons[hor_idx][apex_x]
        else:
            apex_x = np.random.randint(10, ntraces - 10)
            t0 = np.random.randint(30, nsamples - 30)
            
        velocity = np.random.uniform(1500.0, 4500.0)
        base_amplitude = np.random.uniform(0.1, 0.35)
        f0_diff = np.random.uniform(40.0, 90.0)
        wavelet_diff = ricker_wavelet(f0=f0_diff)
        half_w = len(wavelet_diff) // 2
        
        decay_k = np.random.uniform(0.01, 0.05)
        
        dx = 15.0
        dt = 0.004
        time_0 = t0 * dt
        
        for tr in range(ntraces):
            dist_from_apex = abs(tr - apex_x)
            dist = dist_from_apex * dx
            time_x = np.sqrt(time_0**2 + 4 * (dist**2) / (velocity**2))
            t_idx = int(np.round(time_x / dt))
            
            if t_idx >= nsamples:
                continue
                
            amp_decay = np.exp(-decay_k * dist_from_apex)
            trace_amp = base_amplitude * amp_decay
            
            if half_w <= t_idx < nsamples - half_w:
                diffractions[tr, t_idx - half_w : t_idx + half_w + 1] += wavelet_diff * trace_amp
                
    noise = np.random.normal(0, 0.02, gather_size).astype(np.float32)
    full_wavefield = reflections + diffractions + noise
    
    p_x, p_t = patch_size
    g_x, g_t = gather_size
    
    start_x = np.random.randint(0, g_x - p_x) if g_x > p_x else 0
    start_t = np.random.randint(0, g_t - p_t) if g_t > p_t else 0
    
    patch_full = full_wavefield[start_x:start_x+p_x, start_t:start_t+p_t]
    patch_diff = diffractions[start_x:start_x+p_x, start_t:start_t+p_t]
    
    return patch_full, patch_diff, reflections, diffractions, full_wavefield

class SyntheticSeismicDataset(data.Dataset):
    def __init__(self, image_size=(128, 128), virtual_size=5000, mode="demultiple"):
        super().__init__()
        self.image_size = image_size
        self.virtual_size = virtual_size
        self.mode = mode

    def __len__(self):
        return self.virtual_size
    
    def irregular_mask(self, data, rate=0.5):
        n = data.size()[-1]
        mask = torch.torch.zeros(data.size(),dtype=torch.float64)
        v = round(n*rate)
        TM = random.sample(range(n),v)
        mask[:,:,TM]=1
        mask = mask.type(torch.HalfTensor)
        return mask

    def __getitem__(self, index):
        patch_full, patch_diff, _, _, _ = generate_seismic_gather(gather_size=(256, 256), patch_size=self.image_size)
        
        img_data = torch.from_numpy(patch_full).float().unsqueeze(0)
        img_data = img_data / (torch.max(torch.abs(img_data)) + 1e-8)

        if self.mode == "demultiple":
            img_label = torch.from_numpy(patch_diff).float().unsqueeze(0)
            img_label = img_label / (torch.max(torch.abs(img_label)) + 1e-8)
            
            if random.random() > 0.5:
                img_data = img_data * -1.0
                img_label = img_label * -1.0
                
            gain = random.uniform(0.8, 1.2)
            img_data = torch.clamp(img_data * gain, min=-1.0, max=1.0)
            img_label = torch.clamp(img_label * gain, min=-1.0, max=1.0)
            
            return img_data, img_label
        elif self.mode == "interpolation":
            return self.irregular_mask(img_data), img_data
        elif self.mode == "denoising":
            mean = torch.mean(img_data)
            std = torch.std(img_data)
            noise = 0.5*torch.normal(mean, std, size =(img_data.shape[0], img_data.shape[1], img_data.shape[2]))
            img_ = img_data + noise
            return img_, img_data
        else:
            raise ValueError("ERROR MODE")

class RealSeismicDataset(data.Dataset):
    def __init__(self, h5_path, mode="demultiple"):
        super().__init__()
        self.h5_path = h5_path
        self.mode = mode
        self.file = h5py.File(h5_path, 'r')
        self.num_patches = len(self.file.keys()) // 2

    def __len__(self):
        return self.num_patches
    
    def irregular_mask(self, data, rate=0.5):
        n = data.size()[-1]
        mask = torch.torch.zeros(data.size(),dtype=torch.float64)
        v = round(n*rate)
        TM = random.sample(range(n),v)
        mask[:,:,TM]=1
        mask = mask.type(torch.HalfTensor)
        return mask

    def __getitem__(self, index):
        x_data_np = self.file[f'X_{index}'][()]
        y_data_np = self.file[f'Y_{index}'][()]
        
        x_data = torch.from_numpy(x_data_np).float().unsqueeze(0)
        y_data = torch.from_numpy(y_data_np).float().unsqueeze(0)
        
        # A normalizacao DEVE ser atrelada ao max(X) para preservar o ganho global entre X e Y
        max_val = torch.max(torch.abs(x_data)) + 1e-8
        x_data = x_data / max_val
        y_data = y_data / max_val

        if self.mode == "demultiple":
            return x_data, y_data
        elif self.mode == "interpolation":
            return self.irregular_mask(img_data), img_data
        elif self.mode == "denoising":
            mean = torch.mean(img_data)
            std = torch.std(img_data)
            noise = 0.5*torch.normal(mean, std, size =(img_data.shape[0], img_data.shape[1], img_data.shape[2]))
            img_ = img_data + noise
            return img_, img_data
        else:
            raise ValueError("ERROR MODE")

# Redirect Dataset to SyntheticSeismicDataset for backwards compatibility in notebooks
Dataset = SyntheticSeismicDataset

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,       
        mode,
        folder,
        *,
        ema_decay = 0.999,
        image_size = (128,128),
        train_batch_size = 32,
        train_lr = 3e-6,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 5000,
        update_ema_every = 1,
        save_and_sample_every = 10000
    ):
        super().__init__()
        self.model = diffusion_model
        self.mode = mode
        self.folder = folder
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        if folder and folder.endswith('.h5'):
            self.ds = RealSeismicDataset(folder, mode=mode)
        else:
            self.ds = SyntheticSeismicDataset(image_size=image_size, virtual_size=10000, mode=mode)
            
        self.dl = cycle(data.DataLoader(
            self.ds, 
            batch_size = train_batch_size, 
            shuffle=True, 
            pin_memory=True,
            num_workers=8,
            prefetch_factor=2
        ))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        
        results_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', f'results_{self.mode}'))

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.scaler.load_state_dict(data['scaler'])

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps, desc='Treinamento (Loss Hibrida)') as pbar:
            while self.step < self.train_num_steps:
                for i in range(self.gradient_accumulate_every):
                    img = next(self.dl)
                    device = next(self.model.parameters()).device
                    inputs = img[0].to(device)
                    gt = img[1].to(device)
                    
                    with autocast(enabled = self.amp):
                        loss = self.model(inputs, gt)
                        self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    loss_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training_loss_log.txt'))
                    with open(loss_log_path, 'a') as f:
                        f.write(f'{self.step}: {loss.item()}\n')
                
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                pbar.set_postfix(loss=f"{loss.item():.5f}")
                pbar.update(1)

                if self.step % self.update_ema_every == 0:
                    self.step_ema()
                if self.step > 0 and self.step % 5000 == 0:
                    milestone_k = f"{self.step // 1000}k"
                    self.save(milestone_k)
                    pbar.write(f"[*] Checkpoint de seguranca salvo: model-{milestone_k}.pt")
                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every
                    inputs_ = torch.unsqueeze(inputs[0], dim=0)
                    if self.mode == "interpolation":
                        gt_ = torch.unsqueeze(gt[0], dim=0)
                        all_images = self.ema_model.inference(x_in=gt_, mask=inputs_)
                    else:
                        all_images = self.ema_model.inference(x_in=inputs_)
                    all_images = (all_images + 1) * 0.5
                    utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = 6)
                    self.save(milestone)

                self.step += 1

        print('training completed')
