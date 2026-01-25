import torch
import numpy as np

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, preds, trues = 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total += criterion(out, y).item() * x.size(0)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
    return total / len(loader.dataset), np.vstack(trues), np.vstack(preds) # stacking arrays in sequence vertically 
