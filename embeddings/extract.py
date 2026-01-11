import torch
from torch.utils.data import DataLoader, TensorDataset

def extract_embeddings(model, X, device, batch_size=64):
    model.eval()
    loader = DataLoader(TensorDataset(X), batch_size=batch_size)
    out = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            _, h = model.gru(xb)
            out.append(torch.cat([h[-2], h[-1]], dim=-1).cpu())
    return torch.cat(out)
