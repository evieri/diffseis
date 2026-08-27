import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.diffusion import generate_seismic_gather

def qc_plot():
    print("Generating Hyperbolic Diffraction Data...")
    patch_size = (128, 128)
    patch_full, patch_diff, reflections, diffractions, full_wavefield = generate_seismic_gather(
        gather_size=(256, 256), patch_size=patch_size
    )
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=150)
    
    clip = np.nanpercentile(np.abs(full_wavefield), 98)
    if clip == 0 or np.isnan(clip):
        clip = 0.05
    vmin, vmax = -clip, clip
    
    im0 = axes[0].imshow(reflections.T, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title("1. Reflection-only Gather", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Trace", fontsize=12)
    axes[0].set_ylabel("Time Sample", fontsize=12)
    
    axes[1].imshow(diffractions.T, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title("2. Diffraction-only Gather", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Trace", fontsize=12)
    
    axes[2].imshow(full_wavefield.T, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    axes[2].set_title("3. Full Wavefield", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Trace", fontsize=12)
    
    axes[3].imshow(patch_full.T, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    axes[3].set_title(f"4. Extracted Patch {patch_size}", fontsize=14, fontweight='bold')
    axes[3].set_xlabel("Trace", fontsize=12)
    
    plt.tight_layout()
    
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = os.path.join(results_dir, 'qc_synthetic_dataset.png')
    plt.savefig(save_path, dpi=150)
    print(f"[+] QC visual gerado com sucesso em: {save_path}")

if __name__ == "__main__":
    qc_plot()
