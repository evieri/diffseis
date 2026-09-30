import os
import sys
import json
import shutil
import subprocess
from typing import Tuple, List, Dict
from pydantic import BaseModel, ConfigDict
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.unet import UNet
from src.diffusion import GaussianDiffusion, SyntheticSeismicDataset
from scripts.plot_metrics import parse_logs, plot_dashboard
from src.reporting import DiffSeisReporter

class TrainingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    num_epochs: int = 1500
    steps_per_epoch: int = 100
    batch_size: int = 4
    lr: float = 2e-4
    timesteps: int = 50
    image_size: int = 128
    metrics_path: str
    dashboard_path: str
    ckpt_dir: str

class EpochMetricRecord(BaseModel):
    epoch: int
    total_loss: float
    mse_loss: float
    pearson_penalty: float
    rms_constraint: float
    lr: float

def create_config(reporter: DiffSeisReporter) -> TrainingConfig:
    return TrainingConfig(
        metrics_path=os.path.join(reporter.run_dir, 'training_metrics.json'),
        dashboard_path=os.path.join(reporter.run_dir, 'training_dashboard.png'),
        ckpt_dir=os.path.join(reporter.run_dir, 'checkpoints')
    )

def setup_pipeline(cfg: TrainingConfig):
    model = UNet(
        in_channel=2, out_channel=1, inner_channel=64, norm_groups=16,
        channel_mults=(1, 2, 4, 8), attn_res=[], res_blocks=1, dropout=0,
        with_noise_level_emb=True, image_size=cfg.image_size
    ).cuda()
    diffusion = GaussianDiffusion(
        model, image_size=cfg.image_size, mode="demultiple", channels=1,
        timesteps=cfg.timesteps, loss_type='l1'
    ).cuda()
    ds = SyntheticSeismicDataset(image_size=(cfg.image_size, cfg.image_size), virtual_size=100000)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    optimizer = Adam(diffusion.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    return diffusion, dl, optimizer, scheduler

def execute_train_step(diffusion, inputs, gt, optimizer) -> Dict[str, float]:
    loss, loss_dict = diffusion(inputs, gt)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    return loss_dict

def train_single_epoch(diffusion, data_iter, optimizer, steps: int) -> Dict[str, float]:
    accum = {"total_loss": 0.0, "mse_loss": 0.0, "pearson_penalty": 0.0, "rms_constraint": 0.0}
    for _ in range(steps):
        inputs, gt = next(data_iter)
        step_dict = execute_train_step(diffusion, inputs.cuda(), gt.cuda(), optimizer)
        for k in accum:
            accum[k] += step_dict.get(k, 0.0)
    return {k: v / steps for k, v in accum.items()}

def save_json_metrics(metrics: List[EpochMetricRecord], file_path: str) -> None:
    data = [m.model_dump() for m in metrics]
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch

def update_dashboard_figure(cfg: TrainingConfig) -> None:
    try:
        epochs, total, mse, pearson, rms, lr = parse_logs(cfg.metrics_path)
        plot_dashboard(epochs, total, mse, pearson, rms, lr, cfg.dashboard_path)
        
        art_dir = '/petrogeo/home/emmanuel/.gemini/antigravity-ide/brain/06ef30f2-6c50-4c9e-988a-8afa046c3499'
        if os.path.exists(art_dir):
            shutil.copyfile(cfg.dashboard_path, os.path.join(art_dir, 'training_dashboard.png'))
    except Exception as e:
        print(f"Erro ao atualizar dashboard: {e}")

def trigger_inference_pipeline():
    print("\n[!] === Disparando automacao de Inferencia Zero-Shot ===")
    script_path = os.path.join(os.path.dirname(__file__), 'inference_su_pipeline.py')
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except Exception as e:
        print(f"Erro durante a inferencia pós-treino: {e}")

def run_training_loop(diffusion, loader, optimizer, scheduler, cfg: TrainingConfig) -> None:
    data_iter = cycle_loader(loader)
    all_metrics = []
    best_loss = float('inf')
    
    pbar = tqdm(range(1, cfg.num_epochs + 1), desc="Treinamento Overnight c-DDPM")
    try:
        for ep in pbar:
            ep_dict = train_single_epoch(diffusion, data_iter, optimizer, cfg.steps_per_epoch)
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            rec = EpochMetricRecord(epoch=ep, lr=current_lr, **ep_dict)
            all_metrics.append(rec)
            
            # Atualiza o melhor modelo
            if ep_dict["total_loss"] < best_loss:
                best_loss = ep_dict["total_loss"]
                torch.save(diffusion.state_dict(), os.path.join(cfg.ckpt_dir, 'cddpm_best.pt'))
            
            # A cada 25 epocas: atualiza JSON e Dashboard
            if ep % 25 == 0:
                save_json_metrics(all_metrics, cfg.metrics_path)
                update_dashboard_figure(cfg)
                
            # A cada 50 epocas: salva checkpoint historico
            if ep % 50 == 0:
                torch.save(diffusion.state_dict(), os.path.join(cfg.ckpt_dir, f'cddpm_epoch_{ep}.pt'))
                
            pbar.set_postfix(loss=f"{rec.total_loss:.4f}", mse=f"{rec.mse_loss:.4f}")
            
    except KeyboardInterrupt:
        print("\n[!] Treinamento interrompido manualmente.")
    except Exception as e:
        print(f"\n[!] Erro fatal no treinamento: {e}")
    finally:
        print("\n=== Treinamento Finalizado ou Interrompido. Salvando log e iniciando inferência ===")
        save_json_metrics(all_metrics, cfg.metrics_path)
        update_dashboard_figure(cfg)
        trigger_inference_pipeline()

def main():
    reporter = DiffSeisReporter()
    cfg = create_config(reporter)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    
    print(f"=== Iniciando Treinamento Overnight (1500 Epocas) no dir: {reporter.run_dir} ===")
    diffusion, loader, optimizer, scheduler = setup_pipeline(cfg)
    run_training_loop(diffusion, loader, optimizer, scheduler, cfg)

if __name__ == "__main__":
    main()
