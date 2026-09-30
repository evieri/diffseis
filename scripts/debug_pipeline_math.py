import os
import sys
import numpy as np
import obspy
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.unet import UNet
from src.diffusion import GaussianDiffusion, SyntheticSeismicDataset

def load_su_as_numpy(file_path):
    stream = obspy.read(file_path, unpack_trace_headers=True)
    dt = stream[0].stats.delta
    headers = [tr.stats.su.trace_header for tr in stream]
    data = np.stack([tr.data for tr in stream])
    return data, dt, headers

def test_identity_reconstruction(data_matrix, patch_size=(128, 128), stride=(64, 64)):
    ntraces, nsamples = data_matrix.shape
    pt, ps = patch_size
    st, ss = stride
    
    pad_traces = max(0, pt - ntraces)
    pad_samples = max(0, ps - nsamples)
    
    pad_traces += (st - ((ntraces + pad_traces - pt) % st)) % st
    pad_samples += (ss - ((nsamples + pad_samples - ps) % ss)) % ss
    
    padded_data = np.pad(data_matrix, ((0, pad_traces), (0, pad_samples)), mode='constant')
    new_ntraces, new_nsamples = padded_data.shape
    
    reconstructed = np.zeros_like(padded_data)
    counts = np.zeros_like(padded_data)
    
    patches_extracted = 0
    for i in range(0, new_ntraces - pt + 1, st):
        for j in range(0, new_nsamples - ps + 1, ss):
            patch = padded_data[i:i+pt, j:j+ps]
            
            # IDENTITY: No U-Net, just add back
            reconstructed[i:i+pt, j:j+ps] += patch
            counts[i:i+pt, j:j+ps] += 1.0
            patches_extracted += 1
            
    counts[counts == 0] = 1.0
    reconstructed = reconstructed / counts
    
    final_output = reconstructed[:ntraces, :nsamples]
    
    rms_error = np.sqrt(np.mean((data_matrix - final_output)**2))
    max_error = np.max(np.abs(data_matrix - final_output))
    return rms_error, max_error, patches_extracted

def main():
    print("=== INICIANDO DIAGNÓSTICO MATEMÁTICO DO PIPELINE ===")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    marine_path = os.path.join(base_dir, 'data/marine-geom-co100.su')
    
    print("\n1. VERIFICAÇÃO DE ORIENTAÇÃO DE EIXOS")
    if not os.path.exists(marine_path):
        print(f"[!] Arquivo não encontrado: {marine_path}")
        return
        
    data, dt, headers = load_su_as_numpy(marine_path)
    ntraces, nsamples = data.shape
    print(f"   -> Shape do array Obspy: {data.shape} (Eixo 0: Traços, Eixo 1: Amostras)")
    patch_size = (128, 128)
    stride = (64, 64)
    print(f"   -> O fatiamento extrai blocos (Traços, Amostras). O tensor vai para o PyTorch como (B, 1, {patch_size[0]}, {patch_size[1]})")
    
    # Checando o dataset de treino para ver a orientação dele:
    print("\n   [Auditoria] Verificando dataset de treinamento:")
    ds = SyntheticSeismicDataset(image_size=(128, 128), virtual_size=1)
    train_in, train_out = ds[0]
    print(f"   -> Shape do patch de treino (PyTorch Dataset): {train_in.shape}")
    
    print("\n2. AUDITORIA DA MATEMÁTICA DE OVERLAP-ADD (IDENTITY TEST)")
    patch_size = (128, 128)
    stride = (64, 64)
    rms, max_err, num_patches = test_identity_reconstruction(data, patch_size, stride)
    print(f"   -> Patches extraídos no teste: {num_patches}")
    print(f"   -> Erro RMS da Reconstrução Identidade: {rms:.6e}")
    print(f"   -> Erro Máximo (Max Abs Error): {max_err:.6e}")
    if rms > 1e-6:
        print("   [!] A reconstrução falhou o teste de identidade (RMS > 0). O overlap-add está corrompendo os dados!")
    else:
        print("   [+] Teste de identidade perfeito (boxcar overlap puro). Porém, se usarmos predições ML, a ausência de uma janela Hanning/Bartlett pode criar artefatos de borda.")

    print("\n3. AUDITORIA DE NORMALIZAÇÃO E ESCALA DE AMPLITUDE")
    pt, ps = patch_size
    sample_patch = data[0:pt, 0:ps]
    
    print(f"   -> Patch Original (cru):")
    print(f"      Min: {sample_patch.min():.4f}, Max: {sample_patch.max():.4f}, Média: {sample_patch.mean():.4f}, Std: {sample_patch.std():.4f}")
    
    # Simula normalização atual do pipeline
    max_val = np.max(np.abs(data))
    if max_val == 0: max_val = 1.0
    padded_data_norm = data / max_val
    norm_patch = padded_data_norm[0:pt, 0:ps]
    
    print(f"   -> Patch Normalizado para U-Net (dividido pelo MaxAbs global {max_val:.4f}):")
    print(f"      Min: {norm_patch.min():.4f}, Max: {norm_patch.max():.4f}, Média: {norm_patch.mean():.4f}, Std: {norm_patch.std():.4f}")
    
    print("\n4. INSPEÇÃO DOS CABEÇALHOS SEISMIC UNIX (.su)")
    print(f"   -> Total de Traços Originais: {len(headers)}")
    print(f"   -> Header tracl do Traço 0: {getattr(headers[0], 'trace_sequence_number_within_line', 'N/A')}")
    print(f"   -> Header ns do Traço 0: {getattr(headers[0], 'number_of_samples_in_this_trace', 'N/A')} (Correto: {nsamples})")

if __name__ == "__main__":
    main()
