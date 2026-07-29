import os
import numpy as np
import obspy
from tqdm import tqdm

def extract_common_offset(su_file: str, target_offset: int, tol: int = 0) -> np.ndarray:
    """
    Le um arquivo .su, extrai os cabecalhos e filtra apenas os tracos
    que correspondem ao offset alvo (dentro de uma tolerancia).
    Retorna uma matriz 2D (num_traces, num_samples).
    """
    print(f"[*] Lendo {su_file} (isso pode demorar dependendo do tamanho)...")
    stream = obspy.read(su_file, unpack_trace_headers=True)
    
    traces = []
    print(f"[*] Filtrando tracos com offset {target_offset} (+/- {tol})...")
    for tr in tqdm(stream.traces, desc="Buscando offsets"):
        offset = tr.stats.su.trace_header.distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group
        if abs(offset - target_offset) <= tol:
            traces.append(tr.data)
            
    if not traces:
        raise ValueError(f"Nenhum traco encontrado com offset {target_offset}")
        
    # Stack tracos verticalmente: shape = (num_traces, num_samples)
    data_2d = np.stack(traces)
    print(f"[+] Secao Common-Offset extraida com sucesso! Dimensoes: {data_2d.shape}")
    return data_2d

def sliding_window_and_save(data_2d: np.ndarray, output_dir: str, 
                            patch_size=(64, 128), stride=(32, 64)):
    """
    Fatia a matriz 2D em blocos exatos usando uma janela deslizante.
    Aplica normalizacao (Max Absoluto) em cada patch e salva em formato .npy.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_traces, num_samples = data_2d.shape
    pt, ps = patch_size
    st, ss = stride
    
    idx = 0
    print(f"[*] Fatiando dados (Patch: {patch_size}, Stride: {stride})...")
    
    for i in range(0, num_traces - pt + 1, st):
        for j in range(0, num_samples - ps + 1, ss):
            patch = data_2d[i:i+pt, j:j+ps]
            
            # Normalizacao (preservando o sinal e a dinamica relativa dentro do bloco)
            max_val = np.max(np.abs(patch))
            if max_val > 0:
                patch = patch / max_val
                
            np.save(os.path.join(output_dir, f"X_{idx}.npy"), patch.astype(np.float32))
            idx += 1
            
    print(f"[+] Sucesso! {idx} blocos salvos em '{output_dir}'.")

if __name__ == "__main__":
    # --- CONFIGURACOES DO PIPELINE ---
    SU_FILE = "/home/emmanuel/Documentos/Trabalho/diffseis/arquivos/marine-geom-co100.su"
    TARGET_OFFSET = -100
    TOLERANCE = 0
    OUTPUT_DIR = "data/data/"
    
    if os.path.exists(SU_FILE):
        print(f"--- Iniciando Pipeline de Ingestao ---")
        data_2d = extract_common_offset(SU_FILE, TARGET_OFFSET, TOLERANCE)
        sliding_window_and_save(data_2d, OUTPUT_DIR)
        print(f"--- Pipeline Concluido ---")
    else:
        print(f"[!] Arquivo nao encontrado: {SU_FILE}")
        print("[!] Por favor, altere a variavel SU_FILE no final do script prepare_real_data.py para testar.")
