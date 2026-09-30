import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import obspy

def load_su_as_numpy(file_path):
    stream = obspy.read(file_path, unpack_trace_headers=True)
    dt = stream[0].stats.delta
    data = np.stack([tr.data for tr in stream])
    return data, dt

def calculate_symmetric_clip(data: np.ndarray, percentile: float = 98.0) -> float:
    clip = float(np.nanpercentile(np.abs(data), percentile))
    return max(clip, 1e-6) if not np.isnan(clip) else 0.05

def plot_comparison(raw_data, pred_data, dt, out_path, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    titles = [f"{title_prefix}: Campo Total", f"{title_prefix}: Difração Isolada"]
    panels = [raw_data, pred_data]
    
    ntraces, nsamples = raw_data.shape
    extent = [0, ntraces, nsamples * dt, 0] # left, right, bottom, top for imshow
    
    for ax, data, title in zip(axes, panels, titles):
        clip = calculate_symmetric_clip(data, 98.0)
        im = ax.imshow(data.T, cmap='seismic', vmin=-clip, vmax=clip, aspect='auto', extent=extent)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Traço", fontsize=10)
        ax.set_ylabel("Tempo [s]", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_fk_spectrum(raw_bp, pred_bp, raw_marine, pred_marine, dt_bp, dt_marine, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    datasets = [
        (raw_bp, "Dado BP: Campo Total", dt_bp),
        (pred_bp, "Dado BP: Difração Isolada", dt_bp),
        (raw_marine, "Dado Marinho: Campo Total", dt_marine),
        (pred_marine, "Dado Marinho: Difração Isolada", dt_marine)
    ]
    
    ax_flat = axes.flatten()
    
    for ax, (data, title, dt) in zip(ax_flat, datasets):
        ntraces, nsamples = data.shape
        
        # Calculate 2D FFT
        fft2d = np.fft.fft2(data)
        fft2d_shifted = np.fft.fftshift(fft2d)
        
        # Magnitude in dB
        magnitude_db = 20 * np.log10(np.abs(fft2d_shifted) + 1e-8)
        
        # Calculate frequencies for axes
        kx = np.fft.fftshift(np.fft.fftfreq(ntraces, d=1.0)) # spatial frequency (normalized)
        f = np.fft.fftshift(np.fft.fftfreq(nsamples, d=dt))  # temporal frequency (Hz)
        
        extent = [kx[0], kx[-1], f[-1], f[0]]
        
        # Vmax for plotting (clip noise floor)
        vmax = np.max(magnitude_db)
        vmin = vmax - 60 # 60 dB dynamic range
        
        im = ax.imshow(magnitude_db.T, cmap='jet', vmin=vmin, vmax=vmax, aspect='auto', extent=extent)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Frequência Espacial (k)", fontsize=10)
        ax.set_ylabel("Frequência Temporal (Hz)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Magnitude (dB)")
        
        # Limit y-axis to positive frequencies up to Nyquist (or 100 Hz for better visualization)
        ax.set_ylim(min(125, f[-1]), 0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    file_pairs = {
        'basico': ('data/dadoBasico_reflecoes_difracoes_be.su', 'results/separated_diffractions_basico.su', 'results/comparison_basico.png'),
        'bp': ('data/dadoBP_offset_100_be_edited.su', 'results/separated_diffractions_bp.su', 'results/comparison_bp.png'),
        'marine': ('data/marine-geom-co100.su', 'results/separated_diffractions_marine.su', 'results/comparison_marine.png')
    }
    
    loaded_data = {}
    
    for name, (in_raw, in_pred, out_fig) in file_pairs.items():
        raw_path = os.path.join(base_dir, in_raw)
        pred_path = os.path.join(base_dir, in_pred)
        out_path = os.path.join(base_dir, out_fig)
        
        if os.path.exists(raw_path) and os.path.exists(pred_path):
            print(f"[*] Gerando figura de comparacao para: {name}")
            raw_data, dt = load_su_as_numpy(raw_path)
            pred_data, _ = load_su_as_numpy(pred_path)
            
            # Subtrair predição do padding se houver mismatch
            nt_raw, ns_raw = raw_data.shape
            pred_data = pred_data[:nt_raw, :ns_raw]
            
            plot_comparison(raw_data, pred_data, dt, out_path, name.capitalize())
            loaded_data[name] = (raw_data, pred_data, dt)
        else:
            print(f"[!] Arquivos faltando para {name}. Pulei.")

    print("\n[*] Gerando Espectro F-K para BP e Marine...")
    fk_out = os.path.join(base_dir, 'results', 'fk_spectrum_comparison.png')
    
    if 'bp' in loaded_data and 'marine' in loaded_data:
        raw_bp, pred_bp, dt_bp = loaded_data['bp']
        raw_mar, pred_mar, dt_mar = loaded_data['marine']
        plot_fk_spectrum(raw_bp, pred_bp, raw_mar, pred_mar, dt_bp, dt_mar, fk_out)
        print(f"[+] Espectro F-K salvo em: {fk_out}")
    else:
        print("[!] Dados BP ou Marine indisponíveis para o espectro F-K.")

    print("\n[+] Todas as figuras foram geradas em 'results/'.")

if __name__ == "__main__":
    main()
