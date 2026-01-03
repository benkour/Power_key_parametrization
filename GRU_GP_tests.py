# %%
import os
import json
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class GRUModel(nn.Module):
    def __init__(self, input_size=29, hidden_size=64, output_size=11, num_layers=2):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Enabling bidirectionality 
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=self.bidirectional)
        # Mapping the 128 hidden features down to 11 parameters which we want to estimate
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size* self.num_directions),
            nn.Linear(hidden_size* self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out, h_n = self.gru(x)
        forward = h_n[-2,:,:]
        backward = h_n[-1,:,:]
        embedding = torch.cat([forward, backward], dim=-1) 
        # Prediction should be the shape of batch_size  and 11
        return self.mlp(embedding)
    
#%%
DATA_DIR = "ml_data/"

X_list = []
y_list = []

param_keys = None
feature_keys = None
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(DATA_DIR, fname)
    with open(fpath, "r") as f:
        data = json.load(f)
    # -------- 1) check presence of outputs --------
    if "outputs" not in data:
        print(f"[WARN] File {fname} has no 'outputs' key, skipping")
        continue
    outputs = data["outputs"]
    if not isinstance(outputs, dict) or len(outputs) == 0:
        print(f"[WARN] File {fname} has empty or invalid 'outputs', skipping")
        continue
    # -------- 2) build time_keys --------
    try:
        time_keys = sorted(outputs.keys(), key=float)
    except ValueError:
        print(f"[WARN] File {fname} has non-numeric time keys: {list(outputs.keys())[:5]}")
        continue
    if len(time_keys) == 0:
        print(f"[WARN] File {fname} has no time steps, skipping")
        continue
    # -------- 3) params / target vector --------
    if "params" not in data:
        print(f"[WARN] File {fname} has no 'params' key, skipping")
        continue
    params = data["params"]
    if param_keys is None:
        param_keys = sorted(params.keys())
    y_vec = [params[k] for k in param_keys]
    y_list.append(y_vec)
    # -------- 4) feature names --------
    if feature_keys is None:
        sample_step = outputs[time_keys[0]]
        feature_keys = sorted(sample_step.keys())
    # -------- 5) build (T, F) matrix for this file --------
    rows = []
    for t in time_keys:
        step_dict = outputs[t]
        row = [step_dict[k] for k in feature_keys]
        rows.append(row)

    X_sample = np.array(rows, dtype=np.float32)  # shape (T, F)
    X_list.append(X_sample)

# -------- 6) stack to arrays --------
X = np.stack(X_list, axis=0)  # (N, T, F)
y = np.stack(y_list, axis=0)  # (N, P)
X_min = X.min(axis=(0,1), keepdims=True)
X_max = X.max(axis=(0,1), keepdims=True)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("param_keys:", param_keys)
print("first 5 feature_keys:", feature_keys[:5])

# %%
from torch.utils.data import DataLoader, TensorDataset

# Here should be data loading parameters
from sklearn.model_selection import train_test_split
y_min = y.min(axis=0)
y_max = y.max(axis=0)
y_raw_range = y_max - y_min

# Finding the constant paramters before modifying the range array
constant_cols = np.where(y_raw_range == 0.0)[0]

# Creating a safe divisor
y_range = y_raw_range.copy()
y_range[y_range == 0] = 1.0 

# Normalizing
y_norm = (y - y_min) / y_range

# Filtering values 
y_min_filtered = np.delete(y_min, constant_cols)
y_max_filtered = np.delete(y_max, constant_cols)
y_norm_filtered = np.delete(y_norm, constant_cols, axis=1)
param_keys_filtered = [k for i,k in enumerate(param_keys) if i not in constant_cols]

# Splitting the data
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_norm, y_norm_filtered, test_size=0.2, random_state=42
)

def denormalize_y(y_pred_norm, y_min, y_max):
    return y_pred_norm * (y_max - y_min) + y_min

# Converting to tensors
X_train_tensor = torch.from_numpy(X_train_np.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train_np.astype(np.float32))
X_test_tensor = torch.from_numpy(X_test_np.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test_np.astype(np.float32))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_size_filtered = y_train_tensor.shape[1]
model = GRUModel(input_size=29, hidden_size=128, output_size=output_size_filtered, num_layers=2)

#Defining the loss function 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Setting hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Creating the missing loader variables for the loop to work
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
def train_model(model, loader, criterion, optimizer, device):
    model.train() # setting model to training mode
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        # backward and optimize
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 

        running_loss += loss.item() * X_batch.size(0)
    return running_loss / len(loader.dataset)

# %%
def evaluate_model(model, loader, criterion, device):
    model.eval() # setting the model to evaluation mode
    test_loss = 0.0
    preds = []
    truths = []
    with torch.no_grad(): 
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()* X_batch.size(0)
            # The batches should be splitted to the sequence of arrays
            preds.append(outputs.cpu().numpy())
            truths.append(y_batch.cpu().numpy())
    # Concatenating all batches into (N,1) arrays
    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0 )
    return test_loss / len(loader.dataset), truths, preds
# %%
models = []
histories = []
num_targets = y_norm_filtered.shape[1]
for target_idx in range(num_targets):
    print(f"\n=== Training model for parameter #{target_idx} ({param_keys_filtered[target_idx]}) ===")
    # selecting one target column
    y_train_t = y_train_tensor[:, target_idx:target_idx+1]
    y_test_t  = y_test_tensor[:,  target_idx:target_idx+1]
    train_dataset_t = TensorDataset(X_train_tensor, y_train_t)
    test_dataset_t  = TensorDataset(X_test_tensor,  y_test_t)
    train_loader_t = DataLoader(train_dataset_t, batch_size=BATCH_SIZE, shuffle=True)
    test_loader_t  = DataLoader(test_dataset_t,  batch_size=BATCH_SIZE, shuffle=False)

    # Create independent model
    model_t = GRUModel(
        input_size=29,
        hidden_size=64,
        output_size=1,      # using a single parameter
        num_layers=2
    ).to(device)

    optimizer_t = torch.optim.Adam(model_t.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    train_hist = []
    test_hist = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model_t, train_loader_t, criterion, optimizer_t, device)
        train_hist.append(train_loss)
        # evaluate on validation each epoch
        val_loss, _, _ = evaluate_model(model_t,  test_loader_t,  criterion, device)
        test_hist.append(val_loss)
        print(f"Param {target_idx} | Epoch {epoch+1}/{NUM_EPOCHS} "
          f"| Train {train_loss:.4f} | Val {val_loss:.4f}")
    test_loss , tr, pr = evaluate_model(model_t,  test_loader_t,  criterion, device)
    # Flattening with squeeze
    tr = tr.squeeze()
    pr = pr.squeeze()
    plt.plot(tr, pr, 'o')
    # Plot the diagonal line 
    lims = [np.min([plt.xlim(), plt.ylim()]),  np.max([plt.xlim(), plt.ylim()])]
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.title(f"parameter_{target_idx}")
    plt.savefig(f"figuresGRU/{target_idx}.pdf")
    plt.show()
    models.append(model_t)
    histories.append((train_hist, test_hist))

print("\n=== All models trained ===\n")

# %%
# The denormalised plot of last parameter
y_min = y_min_filtered[target_idx]
y_max = y_max_filtered[target_idx]

true_den = tr * (y_max - y_min) + y_min
pred_den = pr * (y_max - y_min) + y_min

plt.figure()
plt.plot(true_den, pred_den, 'o')
lims = [np.min([plt.xlim(), plt.ylim()]),  np.max([plt.xlim(), plt.ylim()])]
plt.plot(lims, lims, 'k-', alpha=0.75)
plt.title(f"parameter_{target_idx} (denormalised)")
plt.xlabel("True (physical)")
plt.ylabel("Predicted (physical)")
plt.savefig(f"figuresGRU/{target_idx}_denorm.pdf")
plt.show()

# %%
# ==== Gaussian Processes ====
import gpytorch

from sklearn.decomposition import PCA
def extract_embeddings(model, X_tensor, device, batch_size=64):
    model.eval()
    embeddings = []
    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=False
    )
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            out, h_n = model.gru(X_batch)
            forward = h_n[-2]
            backward = h_n[-1]
            emb = torch.cat([forward, backward], dim=-1)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


# Using exact inferecne, the simplest GP model
class EmbeddingGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        # Mean function, before seeing the data, assume the function is flat
        # GP learns deviations via the kernel
        self.mean_module = gpytorch.means.ConstantMean() 
        # Covariance module
        self.covar_module = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.MaternKernel(
        nu=2.5,
        ard_num_dims=train_x.shape[1]
    )
)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# %%
gp_models = []

for target_idx, model_t in enumerate(models):
    print(f"\n=== Training GP for parameter {param_keys_filtered[target_idx]} ===")

    # Freeze GRU
    for p in model_t.parameters():
        p.requires_grad = False
    model_t = model_t.to(device)

    # Extract embeddings
    Z_train = extract_embeddings(model_t, X_train_tensor, device)
    Z_test  = extract_embeddings(model_t, X_test_tensor, device)
    
    Z_train = Z_train.float()
    Z_test  = Z_test.float()
    # Normalising the values
    Z_mean = Z_train.mean(dim=0, keepdim=True)
    Z_std  = Z_train.std(dim=0, keepdim=True) + 1e-6

    Z_train = (Z_train - Z_mean) / Z_std
    Z_test  = (Z_test  - Z_mean) / Z_std
    pca = PCA(n_components=20)   
    Z_train = torch.from_numpy(pca.fit_transform(Z_train.numpy())).float()
    Z_test  = torch.from_numpy(pca.transform(Z_test.numpy())).float()



    y_train_gp = y_train_tensor[:, target_idx].cpu()
    y_test_gp  = y_test_tensor[:, target_idx].cpu()

    # Likelihood + GP, model observation noise
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
    )
    gp_model = EmbeddingGP(Z_train, y_train_gp, likelihood)

    gp_model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    for i in range(100):
        optimizer.zero_grad()
        output = gp_model(Z_train)
        loss = -mll(output, y_train_gp) # what is happening here?
        loss.backward()
        optimizer.step()

    gp_model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(gp_model(Z_test))
        mean = pred_dist.mean.numpy()
        std  = pred_dist.stddev.numpy()

    gp_models.append((gp_model, likelihood))

    # Plot GP mean vs truth
    plt.figure()
    plt.errorbar(
        y_test_gp.numpy(),
        mean,
        yerr=2*std,
        fmt='o',
        alpha=0.5
    )
    lims = [
        min(plt.xlim()[0], plt.ylim()[0]),
        max(plt.xlim()[1], plt.ylim()[1])
    ]
    plt.plot(lims, lims, 'k--')
    plt.xlabel("True (normalized)")
    plt.ylabel("GP mean ± 2σ")
    plt.title(f"GP on embeddings: {param_keys_filtered[target_idx]}")
    plt.show()

# %%
