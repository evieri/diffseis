import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches

# Paleta do Gamma
COR_FUNDO = '#F7F3F0'
COR_LINHA = '#201B18'
COR_UNET = '#345956'
COR_TEXTO_UNET = '#F2BE5C'

fig = plt.figure(figsize=(10, 5), facecolor=COR_FUNDO)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

nt, nx = 128, 64
x = np.arange(nx)

# ==========================================
# 1. Física Cinemática do Sismograma (Canal 2)
# ==========================================
sismograma_sujo = np.zeros((nt, nx))
shot_pos = 32 # Tiro centralizado

# Evento 1: Reflexão Primária (Alta energia)
t0_primaria = 30
v_primaria = 1.2
curva_primaria = np.sqrt(t0_primaria**2 + ((x - shot_pos) / v_primaria)**2).astype(int)
for i in range(nx):
    if curva_primaria[i] < nt:
        sismograma_sujo[curva_primaria[i], i] = 1.0

# Evento 2: Múltipla (Alta energia, dobro do tempo, fase invertida)
t0_multipla = t0_primaria * 2 # 60
curva_multipla = np.sqrt(t0_multipla**2 + ((x - shot_pos) / v_primaria)**2).astype(int)
for i in range(nx):
    if curva_multipla[i] < nt:
        sismograma_sujo[curva_multipla[i], i] = -0.8 

# Evento 3: Difração isolada (Baixa energia, ponto deslocado)
t0_difracao = 85
pos_difrador = 45 
v_difracao = 1.5 # Maior velocidade na profundidade deixa a hipérbole mais aberta
curva_difracao = np.sqrt(t0_difracao**2 + ((x - pos_difrador) / v_difracao)**2).astype(int)
for i in range(nx):
    if curva_difracao[i] < nt:
        sismograma_sujo[curva_difracao[i], i] = 0.25 # Amplitude baixa

# Aplica a wavelet e ruído de fundo natural
sismograma_sujo = gaussian_filter(sismograma_sujo, sigma=(1.2, 0.8))
np.random.seed(42)
sismograma_sujo += np.random.randn(nt, nx) * 0.05

# ==========================================
# 2. Ruído Estocástico (Canal 1 e Saída)
# ==========================================
ruido_puro = np.random.randn(nt, nx)

# A U-Net estima o ruído. Visualmente será quase idêntico ao ruido_puro,
# provando que ela não cospe o sismograma limpo, mas sim o erro calculado.
ruido_estimado = ruido_puro * 0.9 + np.random.randn(nt, nx) * 0.1 

# ==========================================
# 3. Renderização Visual do Fluxograma
# ==========================================
ax_ruido = fig.add_axes([0.05, 0.25, 0.15, 0.5])
ax_ruido.imshow(ruido_puro, cmap='Greys', aspect='auto')
ax_ruido.axis('off')
ax_ruido.set_title("Canal 1: $x_t$\n(Ruído Estocástico)", color=COR_LINHA, pad=10)

ax_cond = fig.add_axes([0.30, 0.25, 0.15, 0.5])
ax_cond.imshow(sismograma_sujo, cmap='Greys', aspect='auto')
ax_cond.axis('off')
ax_cond.set_title("Canal 2: $c$\n(Sismograma Condicional)", color=COR_LINHA, pad=10)

ax_out = fig.add_axes([0.80, 0.25, 0.15, 0.5])
ax_out.imshow(ruido_estimado, cmap='Greys', aspect='auto')
ax_out.axis('off')
ax_out.set_title("Saída da U-Net\n(Estimativa de Ruído $\epsilon_\\theta$)", color=COR_LINHA, pad=10)

# Símbolos matemáticos
ax.text(0.24, 0.5, "$\oplus$", fontsize=20, color=COR_LINHA, ha='center', va='center')
ax.text(0.50, 0.5, "$\\longrightarrow$", fontsize=24, color=COR_LINHA, ha='center', va='center')
ax.text(0.765, 0.5, "$\\longrightarrow$", fontsize=24, color=COR_LINHA, ha='center', va='center')

# Bloco da U-Net Centralizado
unet_rect = patches.Rectangle((0.55, 0.35), 0.18, 0.3, facecolor=COR_UNET, edgecolor=COR_LINHA, lw=2)
ax.add_patch(unet_rect)
ax.text(0.64, 0.5, "U-Net\n(in_channels=2)", color=COR_TEXTO_UNET, fontsize=12, fontweight='bold', ha='center', va='center')

plt.savefig('diagrama_condicionamento_fisico.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
print("Diagrama fisicamente coerente gerado com sucesso!")