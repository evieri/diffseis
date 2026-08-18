import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.diffusion import GaussianDiffusion
from src.unet import UNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_size = (64, 128)
    model = UNet(
        in_channel=2,
        out_channel=1
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        mode="demultiple",
        channels=1,
        image_size=image_size,
        timesteps=2000,
        loss_type='l2',
    ).to(device)
    
    checkpoint = torch.load('results_demultiple/model-final.pt', map_location=device, weights_only=False)
    diffusion.load_state_dict(checkpoint['ema'])
    diffusion.eval()
    
    x_in_np = np.load('data/data/X_0.npy')
    y_true_np = np.load('data/labels/y_0.npy')
    
    x_in_t = torch.from_numpy(x_in_np).float().unsqueeze(0).unsqueeze(0).to(device)
    x_in_t = x_in_t / (torch.max(torch.abs(x_in_t)) + 1e-8)
    
    y_true_t = torch.from_numpy(y_true_np).float().unsqueeze(0).unsqueeze(0).to(device)
    y_true_t = y_true_t / (torch.max(torch.abs(y_true_t)) + 1e-8)
    
    with torch.no_grad():
        y_pred_t = diffusion.inference(x_in=x_in_t)
        
    y_pred_np = y_pred_t.squeeze().cpu().numpy()
    x_in_np_norm = x_in_t.squeeze().cpu().numpy()
    y_true_np_norm = y_true_t.squeeze().cpu().numpy()
    
    noise_pred = x_in_np_norm - y_pred_np
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    vmax = max(np.max(np.abs(x_in_np_norm)), np.max(np.abs(y_true_np_norm)))
    
    axes[0].imshow(x_in_np_norm.T, cmap='gray', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[0].set_title('Dado Sujo (Entrada)')
    
    axes[1].imshow(y_true_np_norm.T, cmap='gray', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Gabarito (Reflexões Reais)')
    
    axes[2].imshow(noise_pred.T, cmap='gray', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Ruído Previsto')
    
    axes[3].imshow(y_pred_np.T, cmap='gray', aspect='auto', vmin=-vmax, vmax=vmax)
    axes[3].set_title('Dado Limpo (Saída Final)')
    
    import datetime
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_png = f"audit_dashboard_{timestamp}.png"
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    print(f"{output_png} gerado com sucesso!")

if __name__ == '__main__':
    main()
