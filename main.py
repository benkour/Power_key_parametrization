from config import *

from data.loading import load_json_timeseries
from data.preprocessing import normalize_X, normalize_y

from models.gru import GRUModel
from training.loops import train_epoch, eval_epoch

from embeddings.extract import extract_embeddings
from gp.gp import EmbeddingGP

from plotting.gp_plots import (
    plot_gp_mean_only,
    plot_gp_with_2sigma,
    plot_gp_sqrt_uncertainty
)

import torch
import gpytorch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def main():
    # ===============================
    # 1. LOAD + NORMALIZE DATA
    # ===============================
    X, y, param_keys, feature_keys = load_json_timeseries(DATA_DIR)

    X = normalize_X(X)
    y_norm, y_min, y_max, mask = normalize_y(y)

    param_keys_filtered = [k for k, m in zip(param_keys, mask) if m]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y_norm, test_size=0.2, random_state=42
    )

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X.shape[2]

    # ===============================
    # 2. TRAIN GRU MODELS
    # ===============================
    gru_models = []

    for i, name in enumerate(param_keys_filtered):
        print(f"\n=== Training GRU for {name} ===")

        train_loader = DataLoader(
            TensorDataset(Xtr, ytr[:, i:i+1]),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        test_loader = DataLoader(
            TensorDataset(Xte, yte[:, i:i+1]),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model = GRUModel(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            output_size=1,
            num_layers=NUM_LAYERS
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.MSELoss()

        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, _, _ = eval_epoch(
                model, test_loader, criterion, device
            )

            print(
                f"Epoch {epoch+1:03d}/{NUM_EPOCHS} "
                f"| Train {train_loss:.4f} | Val {val_loss:.4f}"
            )

        gru_models.append(model)

    # ===============================
    # 3. GAUSSIAN PROCESSES
    # ===============================
    len(param_keys_filtered) == len(gru_models) == ytr.shape[1]
    #for target_idx, model_t in enumerate(gru_models):
    for target_idx, name in enumerate(param_keys_filtered):
        model_t = gru_models[target_idx]
        print(f"\n=== Training GP for {param_keys_filtered[target_idx]} ===")

        # Freeze GRU
        for p in model_t.parameters():
            p.requires_grad = False
        model_t.eval()

        # ---- embeddings ----
        Z_train = extract_embeddings(model_t, Xtr, device)
        Z_test  = extract_embeddings(model_t, Xte, device)

        Z_train = Z_train.float()
        Z_test  = Z_test.float()

        # normalize embeddings
        Z_mean = Z_train.mean(0, keepdim=True)
        Z_std  = Z_train.std(0, keepdim=True) + 1e-6
        Z_train = (Z_train - Z_mean) / Z_std
        Z_test  = (Z_test  - Z_mean) / Z_std

        # PCA
        pca = PCA(n_components=20)
        Z_train = torch.from_numpy(pca.fit_transform(Z_train.numpy())).float()
        Z_test  = torch.from_numpy(pca.transform(Z_test.numpy())).float()

        y_train_gp = ytr[:, target_idx].cpu()
        y_test_gp  = yte[:, target_idx].cpu()

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        )

        gp_model = EmbeddingGP(Z_train, y_train_gp, likelihood)

        gp_model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        for _ in range(100):
            optimizer.zero_grad()
            output = gp_model(Z_train)
            loss = -mll(output, y_train_gp)
            loss.backward()
            optimizer.step()

        # ---- inference ----
        gp_model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = likelihood(gp_model(Z_test))
            mean = pred_dist.mean.numpy()
            std  = pred_dist.stddev.numpy()

        # ===============================
        # 4. PLOTTING
        # ===============================
        y_true = y_test_gp.numpy()

        plot_gp_mean_only(
            y_true, mean, param_keys_filtered[target_idx]
        )

        plot_gp_with_2sigma(
            y_true, mean, std, param_keys_filtered[target_idx]
        )

        plot_gp_sqrt_uncertainty(
            y_true, mean, std, param_keys_filtered[target_idx]
        )


if __name__ == "__main__":
    main()
