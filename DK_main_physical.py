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

    # ===============================
    # 2. TRAIN GRU MODELS
    # ===============================
    gru_models = []

    for i, name in enumerate(param_keys_filtered):
        print(f"\n=== Training GRU for {name} ===")
        
        print("Target y shape:", ytr[:, i:i+1].shape)
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

    print("\nTotal GRU Models:", len(gru_models))

    # ===============================
    # 3. GAUSSIAN PROCESSES
    # ===============================
    # Testing if condition correct
    assert len(gru_models) == len(param_keys_filtered) == ytr.shape[1]
    for target_idx, (name, model_t) in enumerate(zip(param_keys_filtered, gru_models)):
        model_t = gru_models[target_idx]
        print(f"\n=== Training GP for {param_keys_filtered[target_idx]} ===")

        """Freezing the GRU (stop computing gradients) to treat is as a fixed feature extractor.
        GP (modeling the function over a fixed input space) assumes that the inputs are fixed.
        """
        for p in model_t.parameters():
            p.requires_grad = False
        model_t.eval()

        # Extracting embeddings - meaning the most of GRU 
        Z_train = extract_embeddings(model_t, Xtr, device)
        Z_test  = extract_embeddings(model_t, Xte, device)

        print("Raw embeddings train:", Z_train.shape)
        print("Raw embeddings test:", Z_test.shape)

        Z_train = Z_train.float()
        Z_test  = Z_test.float()

        # normalize embeddings
        Z_mean = Z_train.mean(0, keepdim=True)
        Z_std  = Z_train.std(0, keepdim=True) + 1e-6
        Z_train = (Z_train - Z_mean) / Z_std
        Z_test  = (Z_test  - Z_mean) / Z_std

        print("Normalized embeddings train:", Z_train.shape)

        """PCA - reducing the data components. Finds combinations of original 
           features that explain the most variation in the data.
        """
        pca = PCA(n_components=20) # reducing dimensions to 20 from ....
        Z_train = torch.from_numpy(pca.fit_transform(Z_train.numpy())).float()
        Z_test  = torch.from_numpy(pca.transform(Z_test.numpy())).float()

        evr_sum = float(pca.explained_variance_ratio_.sum())
        print(f"[PCA] {name}: kept {pca.n_components_} dims, explained_var_sum={evr_sum:.4f}")
        print(f"[PCA] first 5 EVR: {pca.explained_variance_ratio_[:5]}")



        print("PCA embeddigns train:", Z_train.shape)
        print("PCA embeddigns test:", Z_test.shape)

        y_train_gp = ytr[:, target_idx].cpu()
        y_test_gp  = yte[:, target_idx].cpu()

        # ---- target stats ----
        ytr_np = y_train_gp.numpy()
        yte_np = y_test_gp.numpy()

        print(f"[y stats] {name}")
        print(f"  train: min={ytr_np.min():.4f} max={ytr_np.max():.4f} mean={ytr_np.mean():.4f} std={ytr_np.std():.4f}")
        print(f"  test : min={yte_np.min():.4f} max={yte_np.max():.4f} mean={yte_np.mean():.4f} std={yte_np.std():.4f}")

        print("GP target train:", y_train_gp.shape)
        print("GP target test:", y_test_gp.shape)

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

        # Inference
        gp_model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = likelihood(gp_model(Z_test))
            mean = pred_dist.mean.numpy()
            std  = pred_dist.stddev.numpy()
        print("GP mean shape:", mean.shape)
        print("GP std shape:", std.shape) 

        # ===============================
        # 4. PLOTTING
        # ===============================
        y_true = y_test_gp.numpy()

        # Plotting physical (not normalized data)
        ymin = y_min[target_idx]
        ymax = y_max[target_idx]

        y_true_phys = y_true * (ymax - ymin) + ymin
        mean_phys   = mean   * (ymax - ymin) + ymin
        std_phys    = std    * (ymax - ymin)
        # Normalized
        plot_gp_mean_only(
            y_true, mean, param_keys_filtered[target_idx]
        )
        # Physical
        plot_gp_mean_only(y_true_phys, mean_phys, name + "_physical")

        plot_gp_with_2sigma(
            y_true, mean, std, param_keys_filtered[target_idx]
        )
        plot_gp_sqrt_uncertainty(
            y_true, mean, std, param_keys_filtered[target_idx]
        )

if __name__ == "__main__":
    main()
