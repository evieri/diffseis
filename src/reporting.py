import os
import json
import numpy as np
import torch
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import obspy
from obspy.core import Trace, Stream

class DiffSeisReporter:
    def __init__(self, run_dir_base: str = "results/runs", run_id: str = None, dt: float = 0.004):
        self.dt = dt
        self.run_id = run_id if run_id else datetime.now().strftime("RUN_%Y%m%d_%H%M%S")
        self.run_dir = os.path.abspath(os.path.join(run_dir_base, self.run_id))
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"[DiffSeisReporter] Inicializado diretório da corrida: {self.run_dir}")

    def _ensure_numpy(self, data):
        """Conversão agnóstica de tensores PyTorch para NumPy numpy.ndarray (2D)."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        # Se veio como batch, pega o primeiro item
        if data.ndim == 4: # B, C, H, W
            data = data[0, 0]
        elif data.ndim == 3: # C, H, W
            data = data[0]
        return data.astype(np.float32)

    def _apply_agc(self, data: np.ndarray, window_length_s: float = 0.15) -> np.ndarray:
        """Aplica Automatic Gain Control (AGC) usando janelas RMS."""
        # Tamanho da janela em amostras
        window_samples = int(window_length_s / self.dt)
        if window_samples % 2 == 0:
            window_samples += 1
        
        half_win = window_samples // 2
        agc_data = np.zeros_like(data)
        
        # AGC padrao por traço
        for tr_idx in range(data.shape[0]):
            trace = data[tr_idx]
            squared = trace ** 2
            # Moving average do quadrado
            rms = np.zeros_like(trace)
            for i in range(len(trace)):
                start = max(0, i - half_win)
                end = min(len(trace), i + half_win + 1)
                rms[i] = np.sqrt(np.mean(squared[start:end]) + 1e-8)
            agc_data[tr_idx] = trace / rms
            
        return agc_data

    def _calculate_symmetric_clip(self, data: np.ndarray, percentile: float = 98.0) -> float:
        clip = float(np.nanpercentile(np.abs(data), percentile))
        return max(clip, 1e-6) if not np.isnan(clip) else 0.05

    def plot_trio_qc(self, input_data, pred_data, title="Trio Geofísico de Controle", 
                     filename="trio_qc.png", dataset_name="Desconhecido"):
        """Gera o Trio Obrigatório: Entrada | Difração | Resíduo."""
        # 1. Preparação dos dados
        inp = self._ensure_numpy(input_data)
        prd = self._ensure_numpy(pred_data)
        
        # Previne erro dimensional (recorta se as bordas foram estendidas)
        min_tr = min(inp.shape[0], prd.shape[0])
        min_sm = min(inp.shape[1], prd.shape[1])
        inp = inp[:min_tr, :min_sm]
        prd = prd[:min_tr, :min_sm]
        
        residual = inp - prd
        
        # 2. AGC (Ganho)
        inp_agc = self._apply_agc(inp)
        prd_agc = self._apply_agc(prd)
        res_agc = self._apply_agc(residual)
        
        panels = [inp_agc, prd_agc, res_agc]
        titles = ["Entrada Bruta", "Difração Isolada", "Resíduo (Reflexões)"]
        
        # 3. Plotagem
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
        extent = [0, min_tr, min_sm * self.dt, 0]
        
        for ax, data, p_title in zip(axes, panels, titles):
            clip = self._calculate_symmetric_clip(data, 98.0)
            im = ax.imshow(data.T, cmap='gray_r', vmin=-clip, vmax=clip, aspect='auto', extent=extent)
            ax.set_title(p_title, fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel("Traço", fontsize=12)
            ax.set_ylabel("Tempo [s]", fontsize=12)
            
        # 4. Metadata Stamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_str = f"DiffSeis c-DDPM | Run ID: {self.run_id} | Data: {timestamp} | Dataset: {dataset_name}"
        fig.text(0.5, 0.02, metadata_str, ha='center', fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # 5. Salvamento
        save_path = os.path.join(self.run_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[DiffSeisReporter] Trio QC gerado com sucesso: {save_path}")
        
    def _save_numpy_to_su(self, data: np.ndarray, out_path: str):
        ntraces, nsamples = data.shape
        stream = Stream()
        for tr_idx in range(ntraces):
            tr = Trace(data=data[tr_idx].astype(np.float32))
            tr.stats.delta = self.dt
            stream.append(tr)
        stream.write(out_path, format='SU')

    def save_state(self, input_data, pred_data, metrics: dict, prefix: str = ""):
        """Salva de forma unificada os 3 arquivos .su e o arquivo .json de métricas."""
        inp = self._ensure_numpy(input_data)
        prd = self._ensure_numpy(pred_data)
        
        min_tr = min(inp.shape[0], prd.shape[0])
        min_sm = min(inp.shape[1], prd.shape[1])
        inp = inp[:min_tr, :min_sm]
        prd = prd[:min_tr, :min_sm]
        residual = inp - prd
        
        prefix_str = f"{prefix}_" if prefix else ""
        
        # Salvamento JSON
        metrics_path = os.path.join(self.run_dir, f"{prefix_str}metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        # Salvamento SU
        self._save_numpy_to_su(inp, os.path.join(self.run_dir, f"{prefix_str}full_wavefield.su"))
        self._save_numpy_to_su(prd, os.path.join(self.run_dir, f"{prefix_str}extracted_diffraction.su"))
        self._save_numpy_to_su(residual, os.path.join(self.run_dir, f"{prefix_str}residual_reflection.su"))
        
        print(f"[DiffSeisReporter] Dados e métricas exportados para {self.run_dir}")
