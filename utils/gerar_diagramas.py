import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ==========================================
# Configurações de Estilo (Paleta do Gamma)
# ==========================================
COR_FUNDO = '#F7F3F0'
COR_LINHA = '#201B18'
plt.rcParams['figure.facecolor'] = COR_FUNDO
plt.rcParams['axes.facecolor'] = COR_FUNDO

# Dimensões do tensor da U-Net
nt, nx = 128, 64 

def aplicar_wavelet(matriz):
    """Suaviza as linhas para parecerem frentes de onda sísmicas"""
    return gaussian_filter(matriz, sigma=(1.5, 1.0))

# ==========================================
# 1. Imagem 1: Ruído Estocástico Puro (Estado x_t)
# ==========================================
np.random.seed(42)
ruido_puro = np.random.randn(nt, nx)

fig1, ax1 = plt.subplots(figsize=(3, 6))
ax1.imshow(ruido_puro, cmap='Greys', aspect='auto')
ax1.axis('off')
plt.tight_layout()
plt.savefig('1_ruido_estocastico.png', dpi=300, transparent=True)
plt.close(fig1)

# ==========================================
# 2. Imagem 2: Sismograma Sujo (A Condição c)
# ==========================================
sismograma_sujo = np.zeros((nt, nx))

# Adiciona Reflexões Planas e Múltiplas (Eventos Fortes)
for i in range(nx):
    sismograma_sujo[30, i] = 1.5
    sismograma_sujo[70, i] = -1.2
    
# Adiciona Múltipla Parabólica
for i in range(nx):
    t_mult = int(90 + ((i - 32)**2) / 40)
    if t_mult < nt:
        sismograma_sujo[t_mult, i] = 1.0

# Adiciona Difrações (Eventos Fracos)
for i in range(nx):
    t_dif1 = int(np.sqrt(45**2 + ((i - 20)*1.5)**2))
    t_dif2 = int(np.sqrt(85**2 + ((i - 45)*2.0)**2))
    if t_dif1 < nt: sismograma_sujo[t_dif1, i] = 0.4
    if t_dif2 < nt: sismograma_sujo[t_dif2, i] = 0.3

sismograma_sujo = aplicar_wavelet(sismograma_sujo)
# Contamina com um pouco de ruído de fundo
sismograma_sujo += np.random.randn(nt, nx) * 0.1

fig2, ax2 = plt.subplots(figsize=(3, 6))
ax2.imshow(sismograma_sujo, cmap='Greys', aspect='auto')
ax2.axis('off')
plt.tight_layout()
plt.savefig('2_sismograma_sujo.png', dpi=300, transparent=True)
plt.close(fig2)

# ==========================================
# 3. Imagem 3: O .su Simples (Prova de Conceito)
# ==========================================
poc_simples = np.zeros((nt, nx))

# Apenas uma hipérbole básica e limpa
for i in range(nx):
    t_poc = int(np.sqrt(60**2 + ((i - 32)*1.8)**2))
    if t_poc < nt:
        poc_simples[t_poc, i] = 1.0

poc_simples = aplicar_wavelet(poc_simples)
poc_simples += np.random.randn(nt, nx) * 0.05 # Ruído mínimo

fig3, ax3 = plt.subplots(figsize=(3, 6))
ax3.imshow(poc_simples, cmap='Greys', aspect='auto')
ax3.axis('off')
plt.tight_layout()
plt.savefig('4_poc_simples.png', dpi=300, transparent=True)
plt.close(fig3)

# ==========================================
# 4. Imagem 4: Diagrama do Sanduíche (Condicionamento U-Net)
# ==========================================
fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.axis('off')

# Plotando miniaturas dentro do diagrama
ax_inset1 = fig4.add_axes([0.05, 0.2, 0.2, 0.6]) # Ruído
ax_inset1.imshow(ruido_puro, cmap='Greys', aspect='auto')
ax_inset1.axis('off')
ax_inset1.set_title("Canal 1: Estado (x_t)\nRuído Estocástico", color=COR_LINHA, fontsize=10, pad=10)

ax_inset2 = fig4.add_axes([0.35, 0.2, 0.2, 0.6]) # Condição
ax_inset2.imshow(sismograma_sujo, cmap='Greys', aspect='auto')
ax_inset2.axis('off')
ax_inset2.set_title("Canal 2: Condição (c)\nSismograma Sujo", color=COR_LINHA, fontsize=10, pad=10)

# Sinais matemáticos e U-Net
fig4.text(0.285, 0.5, "$\oplus$", fontsize=24, color=COR_LINHA, ha='center', va='center') # Símbolo de concatenação
fig4.text(0.60, 0.5, "$\\longrightarrow$", fontsize=30, color=COR_LINHA, ha='center', va='center')

# Bloco da U-Net
unet_box = plt.Rectangle((0.68, 0.35), 0.25, 0.3, fill=True, facecolor=COR_LINHA, edgecolor=COR_LINHA, alpha=0.9)
ax4.add_patch(unet_box)
fig4.text(0.805, 0.5, "U-Net\n(in_channels=2)", color=COR_FUNDO, fontsize=12, fontweight='bold', ha='center', va='center')

plt.savefig('3_diagrama_condicionamento.png', dpi=300, transparent=True)
plt.close(fig4)

print("Imagens geradas com sucesso! Verifique os 4 arquivos PNG na sua pasta.")