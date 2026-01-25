from config import *
from data.loading import load_json_timeseries
from data.preprocessing import normalize_X, normalize_y
from training.loops import train_epoch, eval_epoch
from embeddings.extract import extract_embeddings
from gp.gp_gru_general import ExactGPModel, GRUModel, GRUGP
from plotting.gp_plots import (
    plot_gp_mean_only,
    plot_gp_with_2sigma,
    plot_gp_sqrt_uncertainty
)
import torch
import gpytorch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from gp.dkl import GRUEmbedder, DKLGPModel

def main():
    # ===============================
    # 1. LOAD + NORMALIZE DATA
    # ===============================
    X, y, param_keys, feature_keys = load_json_timeseries(DATA_DIR)
    print("Raw X:", X.shape)
    print("Raw y:", y.shape)
    X = normalize_X(X)
    y_norm, y_min, y_max, constant_mask = normalize_y(y)
    print("Normalized X:", X.shape)
    print("Normalized (filtered) y", y_norm.shape)
    param_keys_filtered = [
        k for k, is_const in zip(param_keys, constant_mask) if not is_const
        ]
    print("Number of active parameters:", len(param_keys_filtered))
    """Split is done before any model training, so it shuold not be any further data contamination"""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_norm, test_size=0.2, random_state=42
    )
    print("Train X:", Xtr.shape, "Test X:", Xte.shape)
    print("Train y:", ytr.shape, "Test y:", yte.shape)

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X.shape[2]
    num_targets = ytr.shape[1]
    for j in range(num_targets):
        name = param_keys_filtered[j]
        print(f"\n=== Training Exact DKL for {name} ===")
        # Playing with shapes
        train_y = ytr[:,j].squeeze(-1) # Shape: (N,)
        test_y = yte[:,j].squeeze(-1)  # Shape: (M,)

        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-4)).to(device)

        feature_extractor = GRUModel(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
        #gp = ExactGPModel(train_x=None, train_y=None, likelihood=likelihood)
        # Total model
        total_model = GRUGP(feature_extractor=feature_extractor, likelihood=likelihood, train_x=Xtr, train_y=train_y).to(device)
        total_model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(total_model.parameters(), lr=LR)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, total_model.gp)    
        # Full batch training

        for epoch in range(NUM_EPOCHS):
            # zeroing the gradient
            optimizer.zero_grad()
            output = total_model(Xtr) # full train set
            loss = -mll(output, train_y)
            loss.backward()
            # updating the model parameter
            optimizer.step()
            if (epoch + 1) % max(1, NUM_EPOCHS // 10) == 0 or epoch == 0:
                print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | loss {loss.item():.4f}")

        # Inference
        total_model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = likelihood(total_model(Xte))
            mean = pred_dist.mean.detach().cpu().numpy()
            std = pred_dist.stddev.detach().cpu().numpy()
        y_true = test_y.detach().cpu().numpy()
        plot_gp_mean_only(y_true, mean, name + "_GRUGP")
        plot_gp_with_2sigma(y_true, mean, std, name + "_GRUGP")
        plot_gp_sqrt_uncertainty(y_true, mean, std, name + "_GRUGP")

        
        ymin = float(y_min[j])
        ymax = float(y_max[j])
        y_true_phys = y_true * (ymax - ymin) + ymin
        mean_phys   = mean   * (ymax - ymin) + ymin
        std_phys    = std    * (ymax - ymin)

        plot_gp_mean_only(y_true_phys, mean_phys, name + "_GRUGP_physical")
        plot_gp_with_2sigma(y_true_phys, mean_phys, std_phys, name + "_GRUGP_physical")
        plot_gp_sqrt_uncertainty(y_true_phys, mean_phys, std_phys, name + "_GRUGP_physical")

    print("\nDone.")


if __name__ == "__main__":
    main()