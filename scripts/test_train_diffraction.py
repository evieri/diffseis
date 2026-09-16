import os
import sys
import torch
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.unet import UNet
from src.diffusion import GaussianDiffusion, Trainer

def main():
    print("=== Iniciando Fast Training Test (Overfitting) ===")
    
    # 1. Limpar json antigo para o teste não misturar dados
    metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training_metrics.json'))
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
        
    model = UNet(
        in_channel=2,
        out_channel=1,
        inner_channel=64,
        norm_groups=16,
        channel_mults=(1, 2, 4, 8),
        attn_res=[],
        res_blocks=1,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ).cuda()
    
    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        mode="demultiple",
        channels=1,
        timesteps=50, # Fast timesteps
        loss_type='l1'
    ).cuda()
    
    trainer = Trainer(
        diffusion, 'demultiple',
        'dummy_path',
        train_batch_size=2, # Small batch for T1000
        train_lr=2e-4,
        train_num_steps=3, # Exactly 3 epochs/steps
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=False # fp32 strict as requested
    )
    
    # Treinar 3 steps
    trainer.train()
    
    print("=== Treinamento Concluido ===")
    print("Invocando plot_metrics.py para gerar dashboard...")
    plot_script = os.path.abspath(os.path.join(os.path.dirname(__file__), 'plot_metrics.py'))
    subprocess.run([sys.executable, plot_script])
    print("Dashboard salvo em results/training_dashboard.png")

if __name__ == "__main__":
    main()
