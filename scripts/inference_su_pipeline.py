import os
import sys
import numpy as np
import torch
import obspy
from obspy.core import Trace, Stream
import copy
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.unet import UNet
from src.diffusion import GaussianDiffusion
from src.reporting import DiffSeisReporter

def load_su_as_numpy(file_path):
    stream = obspy.read(file_path, unpack_trace_headers=True)
    dt = stream[0].stats.delta
    headers = [tr.stats.su.trace_header for tr in stream]
    
    data = []
    for tr in stream:
        data.append(tr.data)
    
    return np.stack(data), dt, headers


def get_2d_hanning_window(patch_size):
    """Gera uma janela de Hanning 2D para suavizar transições de borda."""
    w_x = np.hanning(patch_size[0])
    w_y = np.hanning(patch_size[1])
    return np.outer(w_x, w_y).astype(np.float32)

def process_seismic_data(diffusion_model, data_matrix, patch_size=(128, 128), stride=(64, 64)):
    ntraces, nsamples = data_matrix.shape
    pt, ps = patch_size
    st, ss = stride
    
    # 1. Padding if smaller than patch size
    pad_traces = max(0, pt - ntraces)
    pad_samples = max(0, ps - nsamples)
    
    # Pad to ensure minimum size and also that it perfectly fits the stride
    pad_traces += (st - ((ntraces + pad_traces - pt) % st)) % st
    pad_samples += (ss - ((nsamples + pad_samples - ps) % ss)) % ss
    
    padded_data = np.pad(data_matrix, ((0, pad_traces), (0, pad_samples)), mode='constant')
    new_ntraces, new_nsamples = padded_data.shape
    
    # Reconstructed matrix
    reconstructed = np.zeros_like(padded_data)
    counts = np.zeros_like(padded_data)
    
    # Window
    window = get_2d_hanning_window(patch_size)
    
    # Overlap-add
    for i in tqdm(range(0, new_ntraces - pt + 1, st), desc="Processando Tracos"):
        for j in range(0, new_nsamples - ps + 1, ss):
            patch = padded_data[i:i+pt, j:j+ps]
            
            patch_max = np.max(np.abs(patch))
            if patch_max < 1e-8:
                patch_max = 1.0
                
            patch_norm = patch / patch_max
            
            x_in = torch.from_numpy(patch_norm).float().unsqueeze(0).unsqueeze(0).cuda()
            
            with torch.no_grad():
                out = diffusion_model.p_sample_loop(x_in)
                pred_patch = out[1, 0].cpu().numpy()
            
            # Desnormaliza e aplica janela de Hanning
            pred_patch = pred_patch * patch_max
            pred_patch_windowed = pred_patch * window
            
            reconstructed[i:i+pt, j:j+ps] += pred_patch_windowed
            counts[i:i+pt, j:j+ps] += window
            
    # Normalize overlap
    counts[counts == 0] = 1.0
    reconstructed = reconstructed / counts
    
    # Unpad
    return reconstructed[:ntraces, :nsamples]

def run_inference_on_file(diffusion_model, in_file, reporter: DiffSeisReporter):
    print(f"\n[*] Processando: {in_file}")
    data, dt, headers = load_su_as_numpy(in_file)
    print(f"    Dimensoes originais: {data.shape}")
    
    separated_data = process_seismic_data(diffusion_model, data)
    
    # Gerando plot de QC
    dataset_name = os.path.basename(in_file).replace('.su', '')
    reporter.plot_trio_qc(
        input_data=data,
        pred_data=separated_data,
        title=f"Avaliação Pós-Treino ({dataset_name})",
        filename=f"trio_qc_{dataset_name}.png",
        dataset_name=dataset_name
    )
    
    # Salvando os .su
    reporter.dt = dt  # Garante que o dt exportado bata com o lido
    reporter.save_state(
        input_data=data,
        pred_data=separated_data,
        metrics={"status": "inferred", "dataset": dataset_name},
        prefix=dataset_name
    )
    print(f"[+] Inferência salva com sucesso em: {reporter.run_dir}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'results', 'checkpoints', 'cddpm_best.pt')
    
    if not os.path.exists(model_path):
        print(f"[!] Erro: Modelo treinado nao encontrado em {model_path}")
        sys.exit(1)
        
    print("=== Carregando modelo c-DDPM para inferencia ===")
    model = UNet(
        in_channel=2, out_channel=1, inner_channel=64, norm_groups=16,
        channel_mults=(1, 2, 4, 8), attn_res=[], res_blocks=1, dropout=0,
        with_noise_level_emb=True, image_size=128
    ).cuda()
    
    diffusion = GaussianDiffusion(
        model, image_size=128, mode="demultiple", channels=1,
        timesteps=50, loss_type='l1'
    ).cuda()
    
    diffusion.load_state_dict(torch.load(model_path))
    diffusion.eval()
    
    files_to_process = [
        'data/marine-geom-co100.su',
        'data/dadoBasico_reflecoes_difracoes_be.su',
        'data/dadoBP_offset_100_be_edited.su'
    ]
    
    reporter = DiffSeisReporter()
    for in_f in files_to_process:
        in_path = os.path.join(base_dir, in_f)
        if os.path.exists(in_path):
            run_inference_on_file(diffusion, in_path, reporter)
        else:
            print(f"[!] Arquivo nao encontrado: {in_path}")
