import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    h5_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/real_dataset.h5'))
    
    tr_patches = 6  # 6 * 64 = 384 tracos
    t_patches = 8   # 8 * 128 = 1024 amostras de tempo
    
    data_sujo = np.zeros((tr_patches * 64, t_patches * 128), dtype=np.float32)
    data_limpo = np.zeros((tr_patches * 64, t_patches * 128), dtype=np.float32)
    
    with h5py.File(h5_path, 'r') as f:
        for i in range(tr_patches):
            for j in range(t_patches):
                patch_idx = i * 15 + j
                data_sujo[i*64:(i+1)*64, j*128:(j+1)*128] = f[f'X_{patch_idx}'][:]
                data_limpo[i*64:(i+1)*64, j*128:(j+1)*128] = f[f'Y_{patch_idx}'][:]
                
    print("\n--- Estatisticas Puras do HDF5 ---")
    print("Data I (Sujo):")
    print(f"  Max:  {np.max(data_sujo)}")
    print(f"  Min:  {np.min(data_sujo)}")
    print(f"  Mean: {np.mean(data_sujo)}")
    print(f"  Std:  {np.std(data_sujo)}")
    print(f"  NaNs: {np.isnan(data_sujo).any()}")
    
    print("\nData II (Gabarito Limpo):")
    print(f"  Max:  {np.max(data_limpo)}")
    print(f"  Min:  {np.min(data_limpo)}")
    print(f"  Mean: {np.mean(data_limpo)}")
    print(f"  Std:  {np.std(data_limpo)}")
    print(f"  NaNs: {np.isnan(data_limpo).any()}")
    print("----------------------------------\n")
    
    # Clipping sísmico para revelar refletores fracos e múltiplas (ganho de visualização)
    clip_i = np.percentile(np.abs(data_sujo), 98)
    clip_ii = np.percentile(np.abs(data_limpo), 98)
    # Usa uma escala comum para comparação justa de amplitude
    clip = max(clip_i, clip_ii)
    if clip == 0:
        clip = 0.05
    
    diff = data_sujo - data_limpo
    clip_diff = np.percentile(np.abs(diff), 98)

    fig, axes = plt.subplots(1, 3, figsize=(18, 10), dpi=300)
    
    # 1. Data I (Sujo com Múltiplas)
    im1 = axes[0].imshow(data_sujo.T, cmap='seismic', aspect='auto', vmin=-clip, vmax=clip, interpolation='bilinear')
    axes[0].set_title(f'Data I: Sujo (Com Múltiplas)\n[Clip: ±{clip:.4f}]', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Traços', fontsize=12)
    axes[0].set_ylabel('Amostras de Tempo (dt=4ms)', fontsize=12)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. Data II (Gabarito Limpo)
    im2 = axes[1].imshow(data_limpo.T, cmap='seismic', aspect='auto', vmin=-clip, vmax=clip, interpolation='bilinear')
    axes[1].set_title(f'Data II: Gabarito Limpo (Primárias)\n[Clip: ±{clip:.4f}]', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Traços', fontsize=12)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Diferença (Múltiplas Removidas)
    im3 = axes[2].imshow(diff.T, cmap='seismic', aspect='auto', vmin=-clip, vmax=clip, interpolation='bilinear')
    axes[2].set_title('Diferença: Data I - Data II\n(Múltiplas Puras)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Traços', fontsize=12)
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle("Diagnóstico Visual Sísmico - Hess VTI (HDF5 com Ganho por Percentil)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results_demultiple/debug_raw_h5.png'))
    plt.savefig(plot_path, dpi=300)
    print(f"[+] Plot com ganho sísmico salvo em: {plot_path}")

if __name__ == '__main__':
    main()
