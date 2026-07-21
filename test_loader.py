import torch
from diffusion import Dataset
from torch.utils.data import DataLoader

def test():
    # Instanciando Dataset do modelo (modo demultiple carrega tanto dados quanto labels)
    ds = Dataset(folder="data/", image_size=(64, 128), mode="demultiple")
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    print("--- Teste do DataLoader Sintético de Reflexões ---")
    print(f"Total de pares no Dataset: {len(ds)}")

    for batch in dl:
        x_batch, y_batch = batch
        print(f"Shape do batch de Entrada (X): {x_batch.shape}")
        print(f"Shape do batch de Labels (y): {y_batch.shape}")
        print(f"Dimensões confirmadas: [batch_size={x_batch.shape[0]}, canais={x_batch.shape[1]}, altura={x_batch.shape[2]}, largura={x_batch.shape[3]}]")
        break

if __name__ == "__main__":
    test()
