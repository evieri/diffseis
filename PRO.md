# PRO-v2.md: Arquitetura de Modelos e Prompting Estratégico (DiffSeis)

Este documento dita a **estratégia de divisão de trabalho entre modelos** no seu setup OpenCode, utilizando os modelos disponíveis nas suas contas Google AI Studio e Groq.

---

## 1. ESTRATÉGIA DE ARQUITETURA DE MODELOS

### 🧠 Raciocínio Profundo (Arquitetura, Física e Debug)
- **Modelo:** `Gemini 3.1 Pro Preview Custom Tools`
- **Quando usar:**
  - Análise matemática complexa (ex: impacto da correlação de Pearson na perda sísmica).
  - Revisão da arquitetura U-Net e skip connections.
  - Depuração de NaNs e underflow de gradientes.
  - Análise do espectro F-K via FFT.
- **Por que:** É o topo de linha da sua lista para raciocínio científico e contexto gigante.

### ⚡ Codificação Rápida (Implementação e Testes)
- **Modelo:** `Gemini 3.7 Flash`
- **Quando usar:**
  - Geração de classes e funções PyTorch.
  - Escrita de DataLoaders, loops de treino e scripts de plotagem.
  - Refatoração rápida e correções de sintaxe.
  - Criação de testes unitários.
- **Por que:** Velocidade de inferência extraordinária (cerca de 3x mais rápido que o Pro) e alta adesão a guias estruturados.

### 🚀 Refatoração Extrema (Via Groq — Gratuito)
- **Modelo:** `GPT OSS 120B` (via Groq)
- **Quando usar:**
  - Tarefas que exigem velocidade máxima sem custo.
  - Refatoração de scripts inteiros.
  - Geração de boilerplate e utilitários simples.
- **Por que:** É o maior modelo disponível na sua lista da Groq, com excelente custo-benefício em lógica de programação.

---

## 2. FLUXO DE TRABALHO RECOMENDADO (MODO PLAN + BUILD)

Sempre que for iniciar uma tarefa grande (como o Bloco 3), siga este fluxo para evitar alucinações e retrabalho:

1. Cole o prompt no chat do OpenCode.
2. Pressione `Tab` para alternar para o **modo Plan**.
3. O agente criará um plano de ação detalhado **sem modificar nenhum arquivo**.
4. **Revise o plano** — confira se as etapas fazem sentido.
5. Aperte `Tab` novamente para o **modo Build** e autorize a execução.

Isso garante que você saiba exatamente o que vai ser mudado antes de qualquer arquivo ser tocado.

---

## 3. PROMPT DE EXECUÇÃO: BLOCO 3 (VERSÃO CORRIGIDA)

**Instruções:**
- Use o **`Gemini 3.7 Flash`** para executar a codificação das tarefas abaixo (rápido e eficiente).
- Se ocorrerem erros matemáticos (NaNs ou divergência), troque para o **`Gemini 3.1 Pro Preview Custom Tools`** usando `/model` e peça para ele depurar.

### 📋 Copie o prompt abaixo no seu OpenCode:

```text
Aja como um Engenheiro de Software Sênior e Doutor em Geofísica. Nosso objetivo agora é implementar o BLOCO 3 do projeto DiffSeis, focado em Compatibilidade, Estabilização da Loss Híbrida e um Teste de Treinamento Rápido (Overfitting Test).

Temos as seguintes informações diagnósticas:
1. A U-Net em 'src/unet.py' é compatível com patches 128x128 (divisões por 2 do downsampling funcionam sem padding assimétrico).
2. O condicionamento c-DDPM em 'src/diffusion.py' concatena corretamente o campo completo (y) e o ruído (xt) via 'torch.cat([x_cond, x_noisy], dim=1)'.

Execute cirurgicamente as seguintes refatorações e crie o script de teste, seguindo estritamente as regras do 'AGENTS.md':

**TAREFA 1: Refatoração da Loss Híbrida (Estabilidade para Difrações)**
- Localize o cálculo da perda de Correlação de Pearson na Loss Híbrida dentro de 'src/diffusion.py'.
- Aplique a trava matemática anti-NaN: `corr = torch.clamp(corr, min=-0.95, max=0.95)`.
- Verifique se as outras travas estão presentes:
  - `eps = 1e-8` em todas as divisões e raízes.
  - `torch.clamp(pred, min=-50.0, max=50.0)` antes de operações sensíveis.

**TAREFA 2: Criação do Script de Overfitting Test (scripts/test_train_diffraction.py)**
- Crie o script de forma limpa e modular.
- O script deve:
  1. Instanciar o Dataset sintético com gerador procedural de hipérboles (patch 128x128).
  2. Instanciar a U-Net condicional e o Trainer com `amp=False` (FP32 estrito).
  3. Configurar batch_size=4 e rodar por 3 épocas com dados sintéticos.
  4. Salvar as métricas em 'training_metrics.json'.
  5. No final, invocar automaticamente o script de plotagem para gerar 'training_dashboard.png' (com `matplotlib.use('Agg')` no topo).

**TAREFA 3: Validação e Atualização do Plano**
- Garanta que todo código rode em modo headless.
- Atualize a seção do Bloco 3 no 'implementation_plan.md', marcando as tarefas concluídas com '[x]'.

**IMPORTANTE - SEGURANÇA:**
- **NÃO execute `git commit` automaticamente.**
- Ao final, **gere a mensagem de commit sugerida** (em texto) para que eu revise manualmente antes de confirmar.

Mostre-me:
1. O código exato das alterações na Loss.
2. O conteúdo completo do script 'test_train_diffraction.py'.
3. A mensagem de commit sugerida (em texto, para revisão).
