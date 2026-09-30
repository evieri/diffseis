import os
import sys
import json
import shutil
from typing import Tuple, List, Dict
from pydantic import BaseModel, ConfigDict
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
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
    num_epochs: int = 30
    steps_per_epoch: int = 30
    batch_size: int = 4
    lr: float = 2e-4
    timesteps: int = 50
    image_size: int = 128
    metrics_path: str
    dashboard_path: str
    inference_path: str


class EpochMetricRecord(BaseModel):
    epoch: int
    total_loss: float
    mse_loss: float
    pearson_penalty: float
    rms_constraint: float
    lr: float


def create_config(reporter: DiffSeisReporter) -> TrainingConfig:
    """Instancia configuracao tipada para treinamento rapido de convergencia."""
    return TrainingConfig(
        num_epochs=30,
        steps_per_epoch=30,
        batch_size=4,
        lr=2e-4,
        timesteps=50,
        image_size=128,
        metrics_path=os.path.join(reporter.run_dir, 'training_metrics.json'),
        dashboard_path=os.path.join(reporter.run_dir, 'training_dashboard.png'),
        inference_path=os.path.join(reporter.run_dir, 'diffraction_separation_result.png')
    )


def setup_pipeline(cfg: TrainingConfig) -> Tuple[GaussianDiffusion, DataLoader, Adam]:
    """Inicializa arquitetura c-DDPM, dataset sintetico e otimizador FP32."""
    model = UNet(
        in_channel=2, out_channel=1, inner_channel=64, norm_groups=16,
        channel_mults=(1, 2, 4, 8), attn_res=[], res_blocks=1, dropout=0,
        with_noise_level_emb=True, image_size=cfg.image_size
    ).cuda()
    diffusion = GaussianDiffusion(
        model, image_size=cfg.image_size, mode="demultiple", channels=1,
        timesteps=cfg.timesteps, loss_type='l1'
    ).cuda()
    ds = SyntheticSeismicDataset(image_size=(cfg.image_size, cfg.image_size), virtual_size=10000)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    optimizer = Adam(diffusion.parameters(), lr=cfg.lr)
    return diffusion, dl, optimizer


def execute_train_step(
    diffusion: GaussianDiffusion,
    inputs: torch.Tensor,
    gt: torch.Tensor,
    optimizer: Adam
) -> Dict[str, float]:
    """Executa um passo de otimizacao com FP32 estrito e gradient clipping."""
    loss, loss_dict = diffusion(inputs, gt)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    return loss_dict


def train_single_epoch(
    diffusion: GaussianDiffusion,
    data_iter,
    optimizer: Adam,
    steps: int
) -> Dict[str, float]:
    """Executa iteracoes de uma epoca e calcula medias de componentes da loss."""
    accum: Dict[str, float] = {"total_loss": 0.0, "mse_loss": 0.0, "pearson_penalty": 0.0, "rms_constraint": 0.0}
    for _ in range(steps):
        inputs, gt = next(data_iter)
        step_dict = execute_train_step(diffusion, inputs.cuda(), gt.cuda(), optimizer)
        for k in accum:
            accum[k] += step_dict.get(k, 0.0)
    return {k: v / steps for k, v in accum.items()}


def save_json_metrics(metrics: List[EpochMetricRecord], file_path: str) -> None:
    """Persiste historico de metricas tipadas em formato JSON no disco."""
    data = [m.model_dump() for m in metrics]
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def cycle_loader(loader: DataLoader):
    """Gerador ciclico infinito para DataLoader."""
    while True:
        for batch in loader:
            yield batch


def run_training_loop(
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    optimizer: Adam,
    cfg: TrainingConfig
) -> None:
    """Loop principal de treinamento gerenciando epocas e gravacao de metricas."""
    data_iter = cycle_loader(loader)
    all_metrics: List[EpochMetricRecord] = []
    pbar = tqdm(range(1, cfg.num_epochs + 1), desc="Treinamento c-DDPM Difracao")
    for ep in pbar:
        ep_dict = train_single_epoch(diffusion, data_iter, optimizer, cfg.steps_per_epoch)
        rec = EpochMetricRecord(epoch=ep, lr=cfg.lr, **ep_dict)
        all_metrics.append(rec)
        save_json_metrics(all_metrics, cfg.metrics_path)
        pbar.set_postfix(total_loss=f"{rec.total_loss:.4f}", mse=f"{rec.mse_loss:.4f}")





def run_inference_evaluation(diffusion: GaussianDiffusion, cfg: TrainingConfig, reporter: DiffSeisReporter) -> None:
    """Gera amostragem no dado de teste e salva resultado sismico de 3 colunas via Reporter."""
    ds = SyntheticSeismicDataset(image_size=(cfg.image_size, cfg.image_size), virtual_size=10)
    test_in, test_tgt = ds[0]
    x_in = test_in.unsqueeze(0).cuda()
    with torch.no_grad():
        out = diffusion.p_sample_loop(x_in)
    pred = out[1, 0].cpu().numpy()
    
    reporter.plot_trio_qc(
        input_data=test_in[0].numpy(),
        pred_data=pred,
        title="Avaliação Pós-Treino (Dataset Sintético)",
        filename=os.path.basename(cfg.inference_path),
        dataset_name="Sintético"
    )
    root_copy = os.path.join(os.path.dirname(cfg.metrics_path), 'diffraction_separation_result.png')
    shutil.copyfile(cfg.inference_path, root_copy)
    art_dir = '/petrogeo/home/emmanuel/.gemini/antigravity-ide/brain/06ef30f2-6c50-4c9e-988a-8afa046c3499'
    if os.path.exists(art_dir):
        shutil.copyfile(cfg.inference_path, os.path.join(art_dir, 'diffraction_separation_result.png'))


def update_dashboard_figure(cfg: TrainingConfig) -> None:
    """Atualiza o grafico de metricas com as curvas das epocas treinadas."""
    epochs, total, mse, pearson, rms, lr = parse_logs(cfg.metrics_path)
    plot_dashboard(epochs, total, mse, pearson, rms, lr, cfg.dashboard_path)
    root_copy = os.path.join(os.path.dirname(cfg.metrics_path), 'training_dashboard.png')
    shutil.copyfile(cfg.dashboard_path, root_copy)
    art_dir = '/petrogeo/home/emmanuel/.gemini/antigravity-ide/brain/06ef30f2-6c50-4c9e-988a-8afa046c3499'
    if os.path.exists(art_dir):
        shutil.copyfile(cfg.dashboard_path, os.path.join(art_dir, 'training_dashboard.png'))


def main() -> None:
    """Ponto de entrada: orquestra configuracao, treino, dashboard e inferencia."""
    reporter = DiffSeisReporter()
    cfg = create_config(reporter)
    os.makedirs(os.path.dirname(cfg.dashboard_path), exist_ok=True)
    
    print(f"=== Iniciando Treinamento Rápido no run_dir: {reporter.run_dir} ===")
    diffusion, loader, optimizer = setup_pipeline(cfg)
    run_training_loop(diffusion, loader, optimizer, cfg)
    
    ckpt_path = os.path.join(reporter.run_dir, 'model_30epochs.pt')
    torch.save(diffusion.state_dict(), ckpt_path)
    print("=== Treinamento Concluido com Sucesso ===")
    update_dashboard_figure(cfg)
    run_inference_evaluation(diffusion, cfg, reporter)
    print(f"[+] Graficos salvos em:\n - {cfg.dashboard_path}\n - {cfg.inference_path}")


if __name__ == "__main__":
    main()
