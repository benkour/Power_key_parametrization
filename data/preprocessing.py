import numpy as np
from sklearn.model_selection import train_test_split

def normalize_X(X):
    X_min = X.min(axis=(0,1), keepdims=True)
    X_max = X.max(axis=(0,1), keepdims=True)
    return (X - X_min) / (X_max - X_min + 1e-8)

def normalize_y(y):
    y_min = y.min(axis=0)
    y_max = y.max(axis=0)
    rng = y_max - y_min
    constant = rng == 0
    rng[constant] = 1.0
    y_norm = (y - y_min) / rng
    return y_norm[:, ~constant], y_min[~constant], y_max[~constant], constant

def split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)
