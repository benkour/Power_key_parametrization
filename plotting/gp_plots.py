# plotting/gp_plots.py
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("Mean_only_plots", exist_ok=True)
os.makedirs("TwoSigma_plots", exist_ok=True)
os.makedirs("Sqrt_uncertainty_images", exist_ok=True)


def _setup_axes(y_true, y_pred):
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    pad = 0.05 * (high - low + 1e-6)
    return low - pad, high + pad


def plot_gp_mean_only(y_true, mean, name):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, mean, s=30, alpha=0.7)

    lo, hi = _setup_axes(y_true, mean)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True (normalized)")
    ax.set_ylabel("GP mean")
    ax.set_title(f"GP mean only: {name}")
    ax.grid(alpha=0.3)

    fig.savefig(
        f"Mean_only_plots/{name}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def plot_gp_with_2sigma(y_true, mean, std, name):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.errorbar(
        y_true, mean,
        yerr=2 * std,
        fmt="o",
        alpha=0.5,
        capsize=2
    )

    lo, hi = _setup_axes(y_true, mean)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True (normalized)")
    ax.set_ylabel("GP mean with 2*sigma")
    ax.set_title(f"GP mean with 2*sigma: {name}")
    ax.grid(alpha=0.3)

    fig.savefig(
        f"TwoSigma_plots/{name}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def plot_gp_sqrt_uncertainty(y_true, mean, std, name):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(
        y_true, mean,
        yerr=np.sqrt(std),
        fmt="o",
        alpha=0.5,
        capsize=2
    )
    lo, hi = _setup_axes(y_true, mean)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True (normalized)")
    ax.set_ylabel("GP mean and sqrt(sigma)")
    ax.set_title(f"GP mean and sqrt(sigma): {name}")
    ax.grid(alpha=0.3)

    fig.savefig(
        f"Sqrt_uncertainty_images/{name}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)
