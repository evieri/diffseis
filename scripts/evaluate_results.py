import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.unet import UNet
from src.diffusion import GaussianDiffusion, RealSeismicDataset

def evaluate():
    print("Starting evaluation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = UNet(in_channel=2, out_channel=1)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    diffusion = GaussianDiffusion(
        model,
        mode='demultiple',
        channels=1,
        image_size=(64, 128),
        timesteps=2000,
        loss_type='l2'
    ).to(device)

    ckpt_path = 'results_demultiple/model-final.pt'
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Looking for others...")
        # attempt to find latest if final does not exist
        pass
    else:
        print(f"Loading checkpoint {ckpt_path}")
        data = torch.load(ckpt_path, map_location=device)
        if 'ema' in data:
            diffusion.load_state_dict(data['ema'])
        elif 'model' in data:
            diffusion.load_state_dict(data['model'])

    # Get data
    print("Loading data...")
    dataset = RealSeismicDataset('data/real_dataset.h5', mode='demultiple')
    x_data, y_data = dataset[0] # [1, 64, 128]

    x_input = x_data.unsqueeze(0).to(device) # [1, 1, 64, 128]
    y_target = y_data.numpy().squeeze()      # [64, 128]
    x_dirty = x_data.numpy().squeeze()       # [64, 128]

    # Inference
    print("Running inference...")
    diffusion.eval()
    with torch.no_grad():
        denoised_tensor = diffusion.inference(x_input)
        # denoised_tensor may have concatenated timesteps. We take the last one.
        denoised = denoised_tensor[-1].squeeze().cpu().numpy()

    # Load loss
    loss_path = 'training_loss_log.txt'
    losses = []
    if os.path.exists(loss_path):
        with open(loss_path, 'r') as f:
            for line in f:
                if ':' in line:
                    try:
                        losses.append(float(line.split(':')[1].strip()))
                    except ValueError:
                        pass

    # Calculate metrics
    pearson = np.corrcoef(y_target.flatten(), denoised.flatten())[0, 1]
    
    rms_clean = np.sqrt(np.mean(y_target**2))
    rms_denoised = np.sqrt(np.mean(denoised**2))
    rms_preservation = (rms_denoised / (rms_clean + 1e-8)) * 100

    # Dominant Frequency using FFT
    trace_idx = x_dirty.shape[0] // 2
    tr_dirty = x_dirty[trace_idx, :]
    tr_clean = y_target[trace_idx, :]
    tr_denoised = denoised[trace_idx, :]

    dt = 0.004
    n_samples = len(tr_clean)
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    fft_dirty = np.abs(np.fft.rfft(tr_dirty))
    fft_clean = np.abs(np.fft.rfft(tr_clean))
    fft_denoised = np.abs(np.fft.rfft(tr_denoised))
    
    dom_freq = freqs[np.argmax(fft_denoised)]

    # Plot
    print("Generating plots...")
    fig = plt.figure(figsize=(18, 10), dpi=300)
    
    # Subplot 1: Loss
    ax1 = plt.subplot(2, 3, 1)
    if losses:
        ax1.plot(losses)
    ax1.set_title('Training Loss Curve')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')

    # Subplot 3: Trace overlay
    ax3 = plt.subplot(2, 3, 2)
    t = np.arange(n_samples) * dt
    ax3.plot(t, tr_dirty, label='Dirty', alpha=0.7)
    ax3.plot(t, tr_denoised, label='Denoised', alpha=0.8)
    ax3.plot(t, tr_clean, label='Clean', alpha=0.7, linestyle='--')
    ax3.set_title(f'1D Trace Overlay (Trace {trace_idx})')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.legend()

    # Subplot 4: FFT Amplitude
    ax4 = plt.subplot(2, 3, 3)
    ax4.plot(freqs, fft_dirty, label='Dirty', alpha=0.7)
    ax4.plot(freqs, fft_denoised, label='Denoised', alpha=0.8)
    ax4.plot(freqs, fft_clean, label='Clean', alpha=0.7, linestyle='--')
    ax4.set_title('1D FFT Amplitude Spectrum')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Amplitude')
    ax4.legend()

    # Subplot 2: Seismic Panel
    global_arr = np.concatenate([x_dirty.flatten(), y_target.flatten(), denoised.flatten()])
    clip = np.percentile(np.abs(global_arr), 98)
    vmin, vmax = -clip, clip
    
    ax21 = plt.subplot(2, 3, 4)
    ax21.imshow(x_dirty.T, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax, interpolation='none')
    ax21.set_title('Dirty Data')
    ax21.set_xlabel('Trace')
    ax21.set_ylabel('Time Sample')

    ax22 = plt.subplot(2, 3, 5)
    ax22.imshow(denoised.T, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax, interpolation='none')
    ax22.set_title('Denoised Data')
    ax22.set_xlabel('Trace')

    ax23 = plt.subplot(2, 3, 6)
    ax23.imshow(y_target.T, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax, interpolation='none')
    ax23.set_title('Clean Data')
    ax23.set_xlabel('Trace')

    # Text box
    textstr = (f'Pearson Correlation: {pearson:.4f}\n'
               f'RMS Energy Preservation: {rms_preservation:.2f}%\n'
               f'Dominant Frequency: {dom_freq:.2f} Hz')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.5, 0.02, textstr, fontsize=12, ha='center', bbox=props)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    os.makedirs('results_demultiple', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'results_demultiple/evaluation_panel_{timestamp}.png'
    plt.savefig(save_path, dpi=300)
    print(f'Evaluation saved to {save_path}')

if __name__ == "__main__":
    evaluate()
