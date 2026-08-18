import sys
import os
import torch
import numpy as np
import h5py
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.unet import UNet
from src.diffusion import GaussianDiffusion, Trainer

def compute_rms(matrix):
    return np.sqrt(np.mean(matrix**2))

def compute_dominant_frequency(matrix, dt=0.004):
    ntraces, nsamples = matrix.shape
    freqs = fftfreq(nsamples, d=dt)
    pos_freq_idx = freqs > 0
    freqs = freqs[pos_freq_idx]
    
    avg_amplitude = np.zeros_like(freqs)
    for tr in range(ntraces):
        f = fft(matrix[tr, :])
        avg_amplitude += np.abs(f[pos_freq_idx])
    
    avg_amplitude /= ntraces
    dominant_freq = freqs[np.argmax(avg_amplitude)]
    return dominant_freq

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNet(in_channel=2, out_channel=1)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    diffusion = GaussianDiffusion(
        model,
        mode="demultiple",
        channels=1,
        image_size=(64, 128),
        timesteps=2000,
        loss_type='l2',
    ).to(device)
    
    trainer = Trainer(
        diffusion,
        mode="demultiple",
        folder=None,
        image_size=(64, 128),
        train_batch_size=4,
        train_lr=2e-5,
        train_num_steps=2000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=False
    )
    
    trainer.load("final")
        
    real_data_path = "data/real_dataset.h5"
    with h5py.File(real_data_path, 'r') as f:
        real_patch_np = f['X_15'][:] # shape (64, 128)
        
    max_val = np.max(np.abs(real_patch_np))
    if max_val > 0:
        real_patch_np = real_patch_np / max_val
    
    x_in = torch.from_numpy(real_patch_np).float().unsqueeze(0).unsqueeze(0).to(device)
    
    trainer.ema_model.eval()
    if hasattr(trainer, 'model'):
        trainer.model.eval()
        
    with torch.no_grad():
        denoised_tensor = trainer.ema_model.inference(x_in=x_in)
        
    real_data = x_in.squeeze().cpu().numpy()
    denoised_data = denoised_tensor[-1].squeeze().cpu().numpy()
    
    real_data = real_data * max_val
    denoised_data = denoised_data * max_val
    
    predicted_noise = real_data - denoised_data
    
    # METRICAS
    # 1. Vazamento de Sinal (Correlação de Pearson entre Dado Real e Ruido)
    corr, _ = pearsonr(real_data.flatten(), predicted_noise.flatten())
    
    # 2. Preservação de Energia (RMS)
    rms_real = compute_rms(real_data)
    rms_clean = compute_rms(denoised_data)
    energy_preservation = (rms_clean / rms_real) * 100 if rms_real > 0 else 0
    
    # 3. Espectro Passa-Baixa (Fourier)
    dt = 0.004 # default seismic dt
    freq_real = compute_dominant_frequency(real_data, dt)
    freq_clean = compute_dominant_frequency(denoised_data, dt)
    
    print("\n--- RELATORIO DE AUDITORIA MATEMATICA ---")
    print(f"[METRICA 1] Correlacao de Pearson (Real vs Ruido): {corr:.4f}")
    if abs(corr) > 0.5:
        print(" -> AVALIACAO: Vazamento severo de sinal detectado. A rede esta classificando parte substancial do sinal como ruido.")
    elif abs(corr) > 0.1:
        print(" -> AVALIACAO: Vazamento moderado. Alguma parte do sinal primario ainda esta indo para o painel de ruido.")
    else:
        print(" -> AVALIACAO: Vazamento minimo. Boa ortogonalidade entre sinal e ruido predito.")
        
    print(f"\n[METRICA 2] Energia RMS (Real): {rms_real:.6f}")
    print(f"[METRICA 2] Energia RMS (Limpo): {rms_clean:.6f}")
    print(f"[METRICA 2] Preservacao de Energia: {energy_preservation:.2f}%")
    if energy_preservation < 50.0:
        print(" -> AVALIACAO: Destruicao severa de energia. A rede esta atenuando excessivamente as amplitudes primarias.")
    elif energy_preservation > 150.0:
        print(" -> AVALIACAO: Ganho de energia espurio. A rede esta introduzindo amplificacao nao-fisica.")
    else:
        print(" -> AVALIACAO: Energia global razoavelmente conservada.")
        
    print(f"\n[METRICA 3] Frequencia Dominante (Real): {freq_real:.1f} Hz")
    print(f"[METRICA 3] Frequencia Dominante (Limpo): {freq_clean:.1f} Hz")
    if freq_clean < freq_real * 0.7:
        print(" -> AVALIACAO: Vies passa-baixa detectado. A rede destruiu componentes de alta frequencia.")
    elif freq_clean > freq_real * 1.3:
        print(" -> AVALIACAO: Adicao de chiado/ruido de alta frequencia detectado.")
    else:
        print(" -> AVALIACAO: Espectro de frequencia principal esta preservado.")
        
if __name__ == "__main__":
    main()
