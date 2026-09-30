import os
import sys
import torch

# Adiciona o diretório raiz ao path para importar a src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.unet import UNet
from src.diffusion import GaussianDiffusion

def main():
    print("Iniciando Sanity Check Extremo da Loss...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Instanciar o modelo
    model = UNet(
        in_channel=2,
        out_channel=1
    ).to(device)

    # Vamos sobrescrever p_losses temporariamente apenas para pegar os componentes
    original_p_losses = GaussianDiffusion.p_losses
    
    components = {}
    
    def hooked_p_losses(self, x_cond, x_start, noise=None):
        import numpy as np
        import torch.nn.functional as F
        from src.diffusion import default
        
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t-1],self.sqrt_alphas_cumprod_prev[t],size=b)).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn(x_start.size(), device='cpu').to(x_start.device))
        x_noisy = self.q_sample(x_start=x_start,continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        x_recon = self.denoise_fn(torch.cat([x_cond, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
        loss_mse = F.mse_loss(noise, x_recon)
        
        sqrt_alpha = continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_sq = (1 - continuous_sqrt_alpha_cumprod**2).sqrt().view(-1, 1, 1, 1)
        x_start_pred = (x_noisy - sqrt_one_minus_alpha_sq * x_recon) / sqrt_alpha
        
        predicted_multiple = x_cond - x_start_pred
        
        # Clamp
        x_start_pred = torch.clamp(x_start_pred, min=-50.0, max=50.0)
        predicted_multiple = torch.clamp(predicted_multiple, min=-50.0, max=50.0)
        
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
        
        rms_pred = torch.sqrt((x_start_pred_f**2).mean(dim=[1,2,3]) + eps)
        rms_true = torch.sqrt((x_start_f**2).mean(dim=[1,2,3]) + eps)
        loss_rms = torch.abs(rms_pred - rms_true).mean()
        
        lambda_corr = 0.1
        lambda_rms = 0.05
        loss = loss_mse.float() + lambda_corr * loss_corr + lambda_rms * loss_rms
        
        components['mse'] = loss_mse.item()
        components['corr'] = loss_corr.item()
        components['rms'] = loss_rms.item()
        components['total'] = loss.item()
        components['std_s_mean'] = std_s.mean().item()
        components['std_m_mean'] = std_m.mean().item()
        components['cov_mean'] = cov.mean().item()
        
        return loss

    GaussianDiffusion.p_losses = hooked_p_losses

    diffusion = GaussianDiffusion(
        model,
        mode = "demultiple",
        channels = 1,
        image_size = (64, 128),
        timesteps = 1000,
        loss_type = 'l1'
    ).to(device)

    # 2. Criar tensores de ruído puro e extremo simulando piores cenários
    print("Gerando tensores extremos...")
    # Multiplicador enorme para estressar a capacidade do float32
    inputs = (torch.randn(2, 1, 64, 128) * 100.0).to(device) 
    gt = (torch.randn(2, 1, 64, 128) * 100.0).to(device)
    
    # Enable AMP se necessário (testando as mesmas condições do trainer)
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler(enabled=False) 
    
    # 3. Passar pela rede
    print("Calculando Forward pass com Mixed Precision desabilitado...")
    with autocast(enabled=False):
        loss = diffusion(inputs, gt)
        
    # 4. Backward
    print("Calculando Backward pass...")
    scaler.scale(loss).backward()
    
    scaler.unscale_(torch.optim.SGD(model.parameters(), lr=0.1)) # Unscale before grad check
    
    # 5. Varredura rigorosa
    print("\n--- RESULTADOS DO DIAGNÓSTICO ---")
    
    has_nan_loss = False
    for k, v in components.items():
        print(f"Loss [{k.upper()}]: {v}")
        import math
        if math.isnan(v) or math.isinf(v):
            has_nan_loss = True
            print(f"ALERTA: {k.upper()} contém NaN ou Inf!")
            
    if not has_nan_loss:
        print("Loss components: OK (sem NaNs ou Infs)")
        
    print("\nChecando Gradientes...")
    has_nan_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                has_nan_grad = True
                print(f"ALERTA: Gradiente com NaN/Inf no parâmetro '{name}'!")
                
    if not has_nan_grad:
        print("Gradientes: OK (sem NaNs ou Infs)")
        
    if has_nan_loss or has_nan_grad:
        print("\n[FALHA] O sanity check encontrou instabilidades numéricas.")
        sys.exit(1)
    else:
        print("\n[SUCESSO] A rede suportou o estresse perfeitamente. Nenhuma anomalia numérica.")
        sys.exit(0)

if __name__ == '__main__':
    main()
