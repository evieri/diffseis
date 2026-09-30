import os
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.diffusion import generate_seismic_gather, SeismicGatherData


class QCPlotConfig(BaseModel):
    patch_size: Tuple[int, int] = (128, 128)
    gather_size: Tuple[int, int] = (256, 256)
    dpi: int = 150
    percentile: float = 98.0
    output_filename: str = "qc_synthetic_dataset.png"


class QCExecutionResult(BaseModel):
    success: bool
    output_path: str
    clip_value: float
    gather_shape: Tuple[int, int]
    patch_shape: Tuple[int, int]


def calculate_symmetric_clip(data: np.ndarray, percentile: float = 98.0) -> float:
    """Calcula clipping simétrico no percentil especificado evitando NaN."""
    clip = float(np.nanpercentile(np.abs(data), percentile))
    if clip <= 0.0 or np.isnan(clip):
        clip = 0.05
    return clip


def _render_single_axis(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    is_first: bool = False
) -> None:
    """Renderiza painel sísmico individual no subplot especificado."""
    ax.imshow(matrix.T, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel("Trace", fontsize=11)
    if is_first:
        ax.set_ylabel("Time Sample", fontsize=11)


def render_four_column_qc(
    data: SeismicGatherData,
    clip: float,
    patch_size: Tuple[int, int]
) -> plt.Figure:
    """Plota 4 colunas horizontais com física de reflexões, difrações e patch."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), dpi=150)
    vmin, vmax = -clip, clip
    
    _render_single_axis(axes[0], data.reflections, "1. Pure Reflections", vmin, vmax, is_first=True)
    _render_single_axis(axes[1], data.diffractions, "2. Pure Diffractions (Decay)", vmin, vmax)
    _render_single_axis(axes[2], data.full_wavefield, "3. Full-Wavefield (Refl+Diff+Noise)", vmin, vmax)
    _render_single_axis(axes[3], data.patch_full, f"4. Extracted Patch {patch_size}", vmin, vmax)
    
    plt.tight_layout()
    return fig


def run_qc(config: QCPlotConfig = QCPlotConfig()) -> QCExecutionResult:
    """Executa a geração do dataset sintético e salva a figura de QC a 150 DPI."""
    print("Gerando dados sintéticos (física hiperbólica + segmenting scenario)...")
    data: SeismicGatherData = generate_seismic_gather(
        gather_size=config.gather_size,
        patch_size=config.patch_size
    )
    clip = calculate_symmetric_clip(data.full_wavefield, config.percentile)
    fig = render_four_column_qc(data, clip, config.patch_size)

    root_dir = Path(__file__).resolve().parent.parent
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = root_dir / config.output_filename
    fig.savefig(str(output_path), dpi=config.dpi)
    fig.savefig(str(results_dir / config.output_filename), dpi=config.dpi)
    plt.close(fig)
    print(f"[+] QC Dataset salvo com sucesso: {output_path}")

    return QCExecutionResult(
        success=True,
        output_path=str(output_path),
        clip_value=clip,
        gather_shape=config.gather_size,
        patch_shape=config.patch_size
    )


if __name__ == "__main__":
    result = run_qc()
    print(f"Execução concluída com sucesso: {result.model_dump_json(indent=2)}")

