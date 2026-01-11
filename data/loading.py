import os
import json
import numpy as np


def load_json_timeseries(DATA_DIR):
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
        if not isinstance(data, dict):
            print(f"[WARN] File {fname} root is not dict, skipping")
            continue

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

        X_sample = np.array(rows, dtype=np.float32)  # (T, F)
        X_list.append(X_sample)

    if len(X_list) == 0:
        raise RuntimeError("No valid JSON files found in DATA_DIR")

    X = np.stack(X_list, axis=0)  # (N, T, F)
    y = np.stack(y_list, axis=0)  # (N, P)

    return X, y, param_keys, feature_keys
