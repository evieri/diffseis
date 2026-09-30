import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.reporting import DiffSeisReporter

def test_reporter():
    print("=== Iniciando Teste Unitario: DiffSeisReporter ===")
    
    # Gerando dados sintéticos (100 traços, 600 amostras)
    np.random.seed(42)
    ntraces, nsamples = 100, 600
    
    # Sintetizando um sinal "cru" e um sinal "predito"
    raw_input = np.random.normal(0, 1.0, (ntraces, nsamples))
    pred_diffraction = np.random.normal(0, 0.3, (ntraces, nsamples))
    
    # Criando métricas falsas para o JSON
    mock_metrics = {
        "ortogonalidade": 0.05,
        "pico_do_apice": 0.85,
        "espectro_fk_snr_db": 15.2
    }
    
    # Instanciando
    reporter = DiffSeisReporter()
    
    # 1. Testando o Trio QC
    reporter.plot_trio_qc(
        input_data=raw_input, 
        pred_data=pred_diffraction, 
        title="Teste de Validação - Trio QC (Facade)",
        dataset_name="Sintético de Teste"
    )
    
    # 2. Testando o salvamento unificado
    reporter.save_state(
        input_data=raw_input,
        pred_data=pred_diffraction,
        metrics=mock_metrics
    )
    
    # Verificação simples do filesystem
    expected_files = [
        "trio_qc.png",
        "metrics.json",
        "full_wavefield.su",
        "extracted_diffraction.su",
        "residual_reflection.su"
    ]
    
    all_exist = True
    for f in expected_files:
        path = os.path.join(reporter.run_dir, f)
        if not os.path.exists(path):
            print(f"[FALHOU] Arquivo não encontrado: {path}")
            all_exist = False
        else:
            print(f"[OK] Arquivo verificado: {f}")
            
    if all_exist:
        print("\n[+] TESTE APROVADO! Todos os artefatos foram gerados corretamente.")
    else:
        print("\n[!] TESTE FALHOU.")

if __name__ == "__main__":
    test_reporter()
