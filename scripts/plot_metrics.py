import os
import json
import math
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_mock_log(file_path):
    metrics = []
    epochs = 100
    for epoch in range(1, epochs + 1):
        # Exponential decay simulation
        total_loss = 2.0 * math.exp(-0.05 * epoch) + 0.05 * math.sin(epoch)
        mse_loss = 1.0 * math.exp(-0.04 * epoch) + 0.02 * math.sin(epoch)
        pearson_penalty = 0.5 * math.exp(-0.06 * epoch)
        rms_constraint = 0.5 * math.exp(-0.03 * epoch)
        
        # LR schedule
        lr = 0.001 * math.exp(-0.01 * epoch)
        
        metrics.append({
            "epoch": epoch,
            "total_loss": max(total_loss, 0),
            "mse_loss": max(mse_loss, 0),
            "pearson_penalty": pearson_penalty,
            "rms_constraint": rms_constraint,
            "lr": lr
        })
    
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[+] Mock log gerado em: {file_path}")

def parse_logs(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de log não encontrado: {file_path}")
    
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Formato JSON inválido no arquivo: {file_path}")
            
    epochs = [row.get("epoch", i) for i, row in enumerate(data)]
    total_loss = [row.get("total_loss", float('nan')) for row in data]
    mse_loss = [row.get("mse_loss", float('nan')) for row in data]
    pearson_penalty = [row.get("pearson_penalty", float('nan')) for row in data]
    rms_constraint = [row.get("rms_constraint", float('nan')) for row in data]
    lr = [row.get("lr", float('nan')) for row in data]
    
    return epochs, total_loss, mse_loss, pearson_penalty, rms_constraint, lr

def plot_dashboard(epochs, total_loss, mse_loss, pearson_penalty, rms_constraint, lr, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    
    # Subplot 1: Total Loss vs Epoch
    axes[0].plot(epochs, total_loss, color='#1f77b4', linewidth=2, label="Total Denoising Loss (c-DDPM)")
    axes[0].set_title("Total Denoising Loss (c-DDPM)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Epochs", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    # Subplot 2: Decomposed Losses
    axes[1].plot(epochs, mse_loss, color='#d62728', linewidth=1.5, label="MSE Fidelity Loss")
    axes[1].plot(epochs, pearson_penalty, color='#2ca02c', linewidth=1.5, label="Pearson Correlation Penalty")
    axes[1].plot(epochs, rms_constraint, color='#9467bd', linewidth=1.5, label="RMS Amplitude Constraint")
    axes[1].set_title("Decomposed Losses", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Epochs", fontsize=12)
    axes[1].set_ylabel("Loss Component Value", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    
    # Subplot 3: LR evolution
    axes[2].plot(epochs, lr, color='#ff7f0e', linewidth=2, label="Learning Rate (LR) Schedule")
    axes[2].set_title("Learning Rate Schedule", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Epochs", fontsize=12)
    axes[2].set_ylabel("Learning Rate", fontsize=12)
    axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[+] Painel de treinamento gerado com sucesso em: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Gera o dashboard de métricas de treinamento.")
    parser.add_argument("--mock", action="store_true", help="Gera dados simulados para teste.")
    args = parser.parse_args()
    
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training_metrics.json'))
    output_plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'training_dashboard.png'))
    
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    if args.mock:
        generate_mock_log(log_path)
        
    try:
        epochs, total, mse, pearson, rms, lr = parse_logs(log_path)
        plot_dashboard(epochs, total, mse, pearson, rms, lr, output_plot_path)
    except Exception as e:
        print(f"[-] Erro ao gerar o painel: {e}")
        
if __name__ == "__main__":
    main()
