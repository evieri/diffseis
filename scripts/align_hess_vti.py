import os
import segyio
import numpy as np
import h5py
from scipy.spatial import cKDTree
from scipy.signal import resample
import time
from tqdm import tqdm

def main():
    print("--- Pipeline de Alinhamento Hess VTI ---")
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/hess_vti'))
    
    file_i = os.path.join(data_dir, 'timodel_shot_data_I.segy')
    file_ii_1 = os.path.join(data_dir, 'timodel_shot_data_II_shot001-320.segy')
    file_ii_2 = os.path.join(data_dir, 'timodel_shot_data_II_shot321-720.segy')
    
    out_h5 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/real_dataset.h5'))
    
    subset_size = 10000
    print(f"[1/5] Lendo {subset_size} tracos iniciais do Data I (Sujo)...")
    
    with segyio.open(file_i, "r", ignore_geometry=True) as f_i:
        offsets_i = f_i.attributes(segyio.TraceField.offset)[0:subset_size]
        # Forcar a conversao explcita para array numpy 2D (num_traces, num_samples)
        data_i_subset = np.zeros((subset_size, f_i.samples.size), dtype=np.float32)
        for i in range(subset_size):
            data_i_subset[i] = f_i.trace[i]
            
    print(f"      Data I carregado com sucesso. Matriz: {data_i_subset.shape}")
    print("[2/5] Lendo cabeçalhos Data II (Limpo - Gabarito)...")
    
    with segyio.open(file_ii_1, "r", ignore_geometry=True) as f_ii_1, \
         segyio.open(file_ii_2, "r", ignore_geometry=True) as f_ii_2:
         
        off_1 = f_ii_1.attributes(segyio.TraceField.offset)[:]
        off_2 = f_ii_2.attributes(segyio.TraceField.offset)[:]
        offsets_ii = np.concatenate([off_1, off_2])
        
        print("[3/5] Construindo KDTree para mapeamento por Offset Absoluto...")
        tree = cKDTree(np.abs(offsets_ii).reshape(-1, 1))
        
        _, indices = tree.query(np.abs(offsets_i).reshape(-1, 1))
        
        print("[4/5] Pareando e reamostrando (Resampling dt=6ms para dt=4ms)...")
        # Matriz alinhada explicita (subset_size, 2000)
        data_ii_aligned = np.zeros((subset_size, 2000), dtype=np.float32)
        
        len_off_1 = len(off_1)
        for idx_i, idx_ii in enumerate(tqdm(indices, desc="Resampling traces")):
            if idx_ii < len_off_1:
                tr_ii = f_ii_1.trace[idx_ii]
            else:
                tr_ii = f_ii_2.trace[idx_ii - len_off_1]
                
            # Resample de tr_ii (1332 amostras) para 2000 amostras usando scipy
            tr_ii_resampled = resample(tr_ii, 2000)
            data_ii_aligned[idx_i] = tr_ii_resampled
            
    print("[5/5] Empacotando em blocos (64, 128) e salvando em HDF5...")
    n_patches_tr = subset_size // 64
    n_patches_time = 2000 // 128
    
    with h5py.File(out_h5, 'w') as h5f:
        patch_idx = 0
        for i in range(n_patches_tr):
            start_tr = i * 64
            end_tr = start_tr + 64
            
            for j in range(n_patches_time):
                start_t = j * 128
                end_t = start_t + 128
                
                patch_i = data_i_subset[start_tr:end_tr, start_t:end_t]
                patch_ii = data_ii_aligned[start_tr:end_tr, start_t:end_t]
                
                h5f.create_dataset(f"X_{patch_idx}", data=patch_i, dtype=np.float32)
                h5f.create_dataset(f"Y_{patch_idx}", data=patch_ii, dtype=np.float32)
                patch_idx += 1
                
    print(f"Sucesso! Foram gerados {patch_idx} patches de tamanho (64, 128).")
    print(f"Arquivo HDF5 salvo em: {out_h5}")

if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"Tempo total de execucao: {time.time()-t0:.2f} segundos")
