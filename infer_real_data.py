import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
from diffusion import GaussianDiffusion, Trainer

def main():
    print("--- Iniciando Pipeline de Inferencia em Dado Real ---")
    
    # 1. Configuracao da Arquitetura (Identica ao Treinamento)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo detectado: {device}")
    
    model = UNet(in_channel=2, out_channel=1).to(device)
    diffusion = GaussianDiffusion(
        model,
        mode="demultiple",
        channels=1,
        image_size=(64, 128),
        timesteps=2000,
        loss_type='l2',
    ).to(device)
    
    trainer = Trainer(
        diffusion,
        mode="demultiple",
        folder="data/",
        image_size=(64, 128),
        train_batch_size=4,
        train_lr=2e-5,
        train_num_steps=2000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=torch.cuda.is_available()
    )
    
    # 2. Carga Segura dos Pesos (EMA)
    checkpoint_path = "results_demultiple/model-final.pt"
    if os.path.exists(checkpoint_path):
        print(f"[*] Carregando pesos do checkpoint: {checkpoint_path}")
        trainer.load("final")
    else:
        print(f"[!] AVISO: Checkpoint {checkpoint_path} nao encontrado!")
        print(f"[!] Executando inferencia com pesos aleatorios (apenas para validar o pipeline visual).")
        print(f"[!] Por favor, copie o arquivo .pt da maquina NVIDIA para esta pasta.")
        
    # 3. Coleta do Dado Real
    # Vamos pegar o bloco X_15.npy gerado pelo prepare_real_data.py
    real_data_path = "data/data/X_15.npy"
    if not os.path.exists(real_data_path):
        raise FileNotFoundError(f"Arquivo de dado real nao encontrado: {real_data_path}. Rode o prepare_real_data.py primeiro.")
        
    print(f"[*] Carregando bloco real: {real_data_path}")
    real_patch_np = np.load(real_data_path) # shape (64, 128)
    
    # Prepara o tensor [B, C, H, W] = [1, 1, 64, 128]
    x_in = torch.from_numpy(real_patch_np).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # 4. Injecao de Processamento (Reverse Diffusion via EMA)
    print("[*] Iniciando Reverse Diffusion (Denoising)...")
    with torch.no_grad():
        # O inference da U-Net retorna o dado no range [-1, 1], vamos normalizar para plotar
        denoised_tensor = trainer.ema_model.inference(x_in=x_in)
        
    # A inferencia do diffusion concatena a entrada com a saida no dim=0
    # ret_img shape = (2, 1, 64, 128). Queremos apenas a imagem gerada (index 1 ou -1)
    real_data = x_in.squeeze().cpu().numpy()
    denoised_data = denoised_tensor[-1].squeeze().cpu().numpy()
    
    # 5. Matematica do Ruido
    print("[*] Calculando Ruido Previsto (Real - Limpo)...")
    predicted_noise = real_data - denoised_data
    
    # 6. Painel Geofisico (Plotagem)
    print("[*] Gerando plotagem geofisica...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Define escala de cinza e limites simetricos globais para comparacao justa
    vmax = max(np.max(np.abs(real_data)), np.max(np.abs(denoised_data)))
    vmin = -vmax
    cmap = 'gray'
    
    # [Esquerda] Dado Real
    im0 = axes[0].imshow(real_data.T, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title("Dado Real (Input)")
    axes[0].set_xlabel("Traces")
    axes[0].set_ylabel("Time Samples")
    
    # [Centro] Ruido Previsto
    im1 = axes[1].imshow(predicted_noise.T, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title("Ruido Removido (Diff)")
    axes[1].set_xlabel("Traces")
    
    # [Direita] Dado Limpo
    im2 = axes[2].imshow(denoised_data.T, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[2].set_title("Dado Limpo (Output U-Net)")
    axes[2].set_xlabel("Traces")
    
    plt.colorbar(im2, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.1)
    
    output_png = "inference_real.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] Inferência finalizada com sucesso! Painel salvo em: {output_png}")

if __name__ == "__main__":
    main()
