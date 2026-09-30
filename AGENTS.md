# AGENTS.md — Regras do Projeto DiffSeis

## Estrutura do Repositório
- `src/` — código fonte (unet.py, diffusion.py)
- `scripts/` — scripts utilitários
- `logs/` — métricas e checkpoints

## Comandos
- Treino: `python scripts/train.py`
- Teste rápido: `python scripts/test_train_diffraction.py`

## Regras de Código (OBRIGATÓRIAS)
- Funções com no máximo 20-30 linhas
- Proibido `except Exception` — use exceções específicas
- Retorne dados tipados (Pydantic)

## Estabilidade Numérica (ANTI-NaN)
- FP32 estrito (`amp=False`)
- Gradiente Clipping: `max_norm=1.0`
- `eps=1e-8` em divisões e raízes
- `torch.clamp(pred, min=-50.0, max=50.0)` antes de operações sensíveis
- Correlação de Pearson: `corr = torch.clamp(corr, min=-0.95, max=0.95)`