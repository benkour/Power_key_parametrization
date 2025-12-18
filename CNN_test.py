# %%

# Ok, let's try a simpler CNN
import numpy as np
import torch 
import json 
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
class CNN_Model(nn.Module):
    def __init__(self, input_size=29, output_size=1):
        super(CNN_Model, self).__init__()
        #self.conv1 = nn.Conv1d(input_size, 64, kernel_size=7, padding=3)
        #self.bn1 = nn.BatchNorm1d(64)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # time becomes 336
        )
        # Block 2
        #self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        #self.bn2 = nn.BatchNorm1d(128)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # # Block 3
        #self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        #self.bn3 = nn.BatchNorm1d(128)
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        #MLP
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.transpose(1,2) # batch 29, 672
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        return self.mlp(x)

# %%
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# y Normalization (Safe Method)
y_min = y.min(axis=0)
y_max = y.max(axis=0)
y_raw_range = y_max - y_min

# Identify constants BEFORE modifying range
constant_cols = np.where(y_raw_range == 0.0)[0]

y_range = y_raw_range.copy()
y_range[y_range == 0] = 1.0 
y_norm = (y - y_min) / y_range

# Filter out constant columns
y_min_filtered = np.delete(y_min, constant_cols)
y_max_filtered = np.delete(y_max, constant_cols)
y_norm_filtered = np.delete(y_norm, constant_cols, axis=1)
param_keys_filtered = [k for i,k in enumerate(param_keys) if i not in constant_cols]

# Split Data
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_norm, y_norm_filtered, test_size=0.2, random_state=42
)

# Convert to Tensors
X_train_tensor = torch.from_numpy(X_train_np.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train_np.astype(np.float32))
X_test_tensor = torch.from_numpy(X_test_np.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test_np.astype(np.float32))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 1. Flatten the time series: (873, 672, 29) -> (873, 672*29)
# This is a "brute force" check.
X_train_flat = X_train_np.reshape(X_train_np.shape[0], -1)
X_test_flat = X_test_np.reshape(X_test_np.shape[0], -1)

# 2. Pick the first parameter target
y_train_flat = y_train_np[:, 0]
y_test_flat = y_test_np[:, 0]

print("Training Random Forest Baseline...")
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train_flat, y_train_flat)

preds = rf.predict(X_test_flat)
mse = mean_squared_error(y_test_flat, preds)
r2 = r2_score(y_test_flat, preds)

print(f"Random Forest MSE: {mse:.5f}")
print(f"Random Forest R2 Score: {r2:.5f}")

# %%
# y Normalization (Safe Method)
y_min = y.min(axis=0)
y_max = y.max(axis=0)
y_raw_range = y_max - y_min

# Identify constants BEFORE modifying range
constant_cols = np.where(y_raw_range == 0.0)[0]

# Safe divisor
y_range = y_raw_range.copy()
y_range[y_range == 0] = 1.0 
y_norm = (y - y_min) / y_range

# Filter out constant columns
y_min_filtered = np.delete(y_min, constant_cols)
y_max_filtered = np.delete(y_max, constant_cols)
y_norm_filtered = np.delete(y_norm, constant_cols, axis=1)
param_keys_filtered = [k for i,k in enumerate(param_keys) if i not in constant_cols]

# Split Data
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_norm, y_norm_filtered, test_size=0.2, random_state=42
)

# Convert to Tensors
X_train_tensor = torch.from_numpy(X_train_np.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train_np.astype(np.float32))
X_test_tensor = torch.from_numpy(X_test_np.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test_np.astype(np.float32))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 4. RANDOM FOREST BASELINE CHECK
# ==========================================
print("\n=== Running Random Forest Baseline Check ===")
# Flatten: (N, 672, 29) -> (N, 19488)
X_train_flat = X_train_np.reshape(X_train_np.shape[0], -1)
X_test_flat = X_test_np.reshape(X_test_np.shape[0], -1)

# Check the first parameter only (for speed)
target_idx_rf = 0 
y_train_flat = y_train_np[:, target_idx_rf]
y_test_flat = y_test_np[:, target_idx_rf]

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train_flat, y_train_flat)

preds_rf = rf.predict(X_test_flat)
mse_rf = mean_squared_error(y_test_flat, preds_rf)
r2_rf = r2_score(y_test_flat, preds_rf)

print(f"Target: {param_keys_filtered[target_idx_rf]}")
print(f"Random Forest MSE: {mse_rf:.5f}")
print(f"Random Forest R2 Score: {r2_rf:.5f}")
if r2_rf < 0.1:
    print(">> WARNING: Random Forest failed to find a signal (R2 < 0.1).")
    print(">> Deep Learning is unlikely to work. Check your features.")
else:
    print(">> SUCCESS: Signal detected. Proceeding to CNN training.")

# ==========================================
# 5. TRAINING HELPERS
# ==========================================
def train_model(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
    return running_loss / len(loader.dataset)

def evaluate_model(model, loader, criterion, device):
    model.eval()
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
            preds.append(outputs.cpu().numpy())
            truths.append(y_batch.cpu().numpy())
            
    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0)
    return test_loss / len(loader.dataset), truths, preds

# ==========================================
# 6. CNN TRAINING LOOP
# ==========================================
BATCH_SIZE = 64
NUM_EPOCHS = 70
LEARNING_RATE = 0.001 

models = []
histories = []
num_targets = y_norm_filtered.shape[1]

print("\n=== Starting CNN Training ===")

for target_idx in range(num_targets):
    param_name = param_keys_filtered[target_idx]
    print(f"\n--- Parameter {target_idx}: {param_name} ---")

    # Select single target
    y_train_t = y_train_tensor[:, target_idx:target_idx+1]
    y_test_t  = y_test_tensor[:,  target_idx:target_idx+1]

    # Datasets
    train_dataset_t = TensorDataset(X_train_tensor, y_train_t)
    test_dataset_t  = TensorDataset(X_test_tensor,  y_test_t)
    train_loader_t = DataLoader(train_dataset_t, batch_size=BATCH_SIZE, shuffle=True)
    test_loader_t  = DataLoader(test_dataset_t,  batch_size=BATCH_SIZE, shuffle=False)

    # Initialize CNN
    model_t = CNN_Model(
        input_size=29, 
        output_size=1
    ).to(device)

    optimizer_t = torch.optim.Adam(model_t.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_hist = []
    
    # Training Loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model_t, train_loader_t, criterion, optimizer_t, device)
        train_hist.append(train_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.5f}")

    # Evaluation
    test_loss, tr, pr = evaluate_model(model_t, test_loader_t, criterion, device)
    
    # Flatten for plotting
    tr = tr.squeeze()
    pr = pr.squeeze()

    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(tr, pr, 'o', alpha=0.6, label='Predictions')
    
    # Diagonal line
    #lims = [np.min([plt.xlim(), plt.ylim()]), np.max([plt.xlim(), plt.ylim()])]
    lims = [min(tr.min(), pr.min()), max(tr.max(), pr.max())]
    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Perfect Fit')
    
    plt.xlabel("True (Normalized)")
    plt.ylabel("Predicted (Normalized)")
    plt.title(f"CNN Results: {param_name}")
    plt.legend()
    # plt.savefig(f"figures/cnn_{target_idx}.pdf") 
    plt.show()

    models.append(model_t)
    histories.append(train_hist)

print("\n=== All models trained ===")

# %%