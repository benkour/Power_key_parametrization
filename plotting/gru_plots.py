import os
import numpy as np
import matplotlib.pyplot as plt

def plot_gru_predictions(
        y_true,
        y_pred, 
        param_name,
        out_dir="figuresGRU"
):
    """GRU scatter and diagonal plot"""
    os.makedirs_(out_dir, exist_ok=True)
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    plt.figure(figsize=(6,6))
    plt.plot(y_true, y_pred, 'o', alpha=0.7)

    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())

    plt.plot([low, high], [low, high], 'k-', alpha=0.75)
    plt.xlabel("True (normalised)")
    plt.ylabel("Predicted (normalised)")
    plt.title(param_name)

    plt.grid(alpha=0.3)
    plt.savefig(
        os.path.join(out_dir, f"{param_name}.pdf"),
        bbox_inches="tight"
    )
    plt.close()