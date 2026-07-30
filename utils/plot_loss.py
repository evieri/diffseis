import matplotlib.pyplot as plt

# Lê o arquivo de log que você gerou
losses = []
with open(os.path.join(os.path.dirname(__file__), '../results/training_loss_log.txt'), 'r') as f:
    for line in f:
        if ':' in line:
            try:
                loss_val = float(line.split(':')[1].strip())
                losses.append(loss_val)
            except ValueError:
                continue

# Configuração de Cores da sua Paleta
cor_fundo = '#F7F3F0'
cor_linha = '#201B18'

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(cor_fundo)
ax.set_facecolor(cor_fundo)

# Plota a curva de convergência
ax.plot(losses, color=cor_linha, linewidth=1.5)

# Estilização minimalista e técnica
ax.set_title('Convergência do Treinamento - MSE Loss', color=cor_linha, fontsize=14, fontname='serif', pad=15)
ax.set_xlabel('Iterações (Steps)', color=cor_linha, fontsize=12, fontname='serif')
ax.set_ylabel('Erro (MSE)', color=cor_linha, fontsize=12, fontname='serif')
ax.tick_params(colors=cor_linha)

for spine in ax.spines.values():
    spine.set_edgecolor(cor_linha)
    
ax.grid(True, linestyle='--', alpha=0.3, color=cor_linha)

# Salva a imagem com fundo transparente e alta resolução
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
print("Sucesso! O gráfico foi salvo como 'loss_curve.png' e está pronto para os slides.")