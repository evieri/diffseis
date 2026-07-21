import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from typing import Tuple

def ricker_wavelet(f0: float = 25.0, dt: float = 0.004, length: int = 11) -> np.ndarray:
    """Gera uma wavelet de Ricker para simular pulsos sismicos."""
    t = np.arange(-(length // 2), length // 2 + 1) * dt
    y = (1.0 - 2.0 * (np.pi ** 2) * (f0 ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (f0 ** 2) * (t ** 2))
    return y.astype(np.float32)

def generate_label(shape: Tuple[int, int]) -> np.ndarray:
    """Gera reflexoes geologicas horizontais ou suavemente inclinadas (Label - y)."""
    ntraces, nsamples = shape
    label = np.zeros(shape, dtype=np.float32)
    wavelet = ricker_wavelet()
    half_w = len(wavelet) // 2
    
    # 2 a 4 eventos continuos (refletores)
    num_events = np.random.randint(2, 5)
    for _ in range(num_events):
        start_time = np.random.randint(20, nsamples - 20)
        # Inclinacao suave
        slope = np.random.uniform(-0.15, 0.15)
        amplitude = np.random.uniform(0.7, 1.0)
        
        for tr in range(ntraces):
            t_idx_exact = start_time + slope * tr
            t_idx = int(np.round(t_idx_exact))
            
            # Convolucao com Ricker inserindo no indice
            if half_w <= t_idx < nsamples - half_w:
                label[tr, t_idx - half_w : t_idx + half_w + 1] += wavelet * amplitude
                
    return label

def generate_input(label: np.ndarray) -> np.ndarray:
    """Gera matriz ruidosa somando reflexoes (multiplas) e ruido Gaussiano (Input - X)."""
    ntraces, nsamples = label.shape
    input_data = label.copy()
    
    # Wavelet de frequencia ligeiramente inferior para multiplas
    wavelet = ricker_wavelet(f0=15.0)
    half_w = len(wavelet) // 2
    
    # Ruido Coerente: 2 a 5 multiplas/eventos inclinados cruzando
    num_multiples = np.random.randint(2, 6)
    for _ in range(num_multiples):
        start_time = np.random.randint(0, nsamples)
        # Inclinacao forte ou curvada
        slope = np.random.uniform(-0.8, 0.8)
        curvature = np.random.uniform(-0.005, 0.005)
        amplitude = np.random.uniform(0.3, 0.7)
        
        for tr in range(ntraces):
            t_idx_exact = start_time + slope * tr + curvature * (tr ** 2)
            t_idx = int(np.round(t_idx_exact))
            
            if half_w <= t_idx < nsamples - half_w:
                input_data[tr, t_idx - half_w : t_idx + half_w + 1] += wavelet * amplitude
                
    # Ruido Gaussiano leve de fundo
    noise = np.random.normal(0, 0.03, label.shape).astype(np.float32)
    input_data += noise
    
    return input_data

def main() -> None:
    shape = (64, 128)
    num_samples = 100
    
    import os
    os.makedirs('data/labels', exist_ok=True)
    os.makedirs('data/data', exist_ok=True)
    
    for i in range(num_samples):
        # 1. Geracao
        y_label = generate_label(shape)
        X_input = generate_input(y_label)
        
        # 2. Salvamento (arrays binarios NPY)
        np.save(f'data/data/{i}.npy', X_input)
        np.save(f'data/labels/{i}.npy', y_label)
        
        # Plot apenas do primeiro sample como validacao
        if i == 0:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            vmax = max(np.max(np.abs(X_input)), np.max(np.abs(y_label)))
            
            axes[0].imshow(X_input.T, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
            axes[0].set_title('Input Ruidoso (X)\n(Reflexoes + Multiplas + Gaussiano)')
            axes[0].set_xlabel('Traces (Offset)')
            axes[0].set_ylabel('Tempo (Samples)')
            
            axes[1].imshow(y_label.T, cmap='seismic', aspect='auto', vmin=-vmax, vmax=vmax)
            axes[1].set_title('Label Limpo (y)\n(Reflexoes Primarias)')
            axes[1].set_xlabel('Traces (Offset)')
            
            plt.tight_layout()
            plt.savefig('sample_01.png', dpi=300)
            plt.close()

    print(f"Sucesso! {num_samples} arrays salvos em data/data/ e data/labels/.")

if __name__ == '__main__':
    main()
