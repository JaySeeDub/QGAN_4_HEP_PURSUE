"""
Utility functions for KDE sampling, SMOTENC oversampling, noise layers, and
differentiable loss components used in physics-informed GAN training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTENC
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


# ============================================================
#  1. KDE FEATURE DISTRIBUTIONS
# ============================================================

FEATURE_LABELS = [
    r"$\eta$", r"Mass", r"$p_T$", r"$\Delta R$",
    r"$\langle \Delta R \rangle$", r"$\sigma_{\Delta R}$",
    r"$\langle \mathrm{Pixel} \rangle$", r"$\sigma_{\mathrm{Pixel}}$"
]


def compute_kde_fits(dataset, verbose: bool = False):
    """
    Compute KDE fits for each feature in the dataset.

    Args:
        dataset: PyTorch Dataset, returns (image, feature_vector)
        verbose: If True, plots each KDE distribution.

    Returns:
        dict[label -> gaussian_kde]
    """
    features = torch.stack([dataset[i][1] for i in range(len(dataset))])
    kde_fits = {}

    if verbose:
        fig, axs = plt.subplots(2, 4, figsize=(20, 6))
        axs = axs.flatten()

    for idx, label in enumerate(FEATURE_LABELS):
        data = features[:, idx].cpu().numpy()
        kde = gaussian_kde(data)
        kde_fits[label] = kde

        if verbose:
            ax = axs[idx]
            ax.hist(data, bins=50, density=True, alpha=0.4, edgecolor="black")
            x = np.linspace(data.min(), data.max(), 400)
            ax.plot(x, kde(x), color="green")
            ax.set_title(label)
            ax.grid(True)

    if verbose:
        plt.tight_layout()
        plt.show()

    return kde_fits


def sample_from_kde(kde_fits: dict, num_samples: int) -> torch.Tensor:
    """
    Sample from KDE distributions for all 8 features.

    Returns:
        Tensor shape (num_samples, 8)
    """
    samples = []

    for label in FEATURE_LABELS:
        kde = kde_fits[label]
        x = kde.resample(num_samples).T.squeeze()
        samples.append(x)

    arr = np.stack(samples, axis=1)
    return torch.tensor(arr, dtype=torch.float32)


# ============================================================
#  2. SMOTENC OVERSAMPLING
# ============================================================

def smotenc_generate(
    data: np.ndarray,
    n_samples: int,
    categorical_features=None,
    batch_size: int = 300_000,
    random_state: int = 42
):
    """
    Generate synthetic samples using SMOTENC, but in memory-safe batches.

    Args:
        data: numpy array, shape (N, F)
        n_samples: total synthetic samples desired
        categorical_features: list of categorical column indices
        batch_size: size per SMOTE run

    Returns:
        numpy array, shape (n_samples, F)
    """
    if categorical_features is None:
        categorical_features = []

    data = np.asarray(data)
    n_features = data.shape[1]

    smote = SMOTENC(
        categorical_features=categorical_features,
        random_state=random_state,
        k_neighbors=3
    )

    generated = np.empty((n_samples, n_features))
    remaining = n_samples
    start = 0

    while remaining > 0:
        step = min(batch_size, remaining)

        # pad dummy points to satisfy SMOTENC
        dummy = np.full((step, n_features), -999)
        labels = np.concatenate([np.ones(len(data)), np.zeros(len(dummy))])
        X = np.vstack([data, dummy])

        X_res, y_res = smote.fit_resample(X, labels)

        synth = X_res[y_res == 1]
        synth = synth[len(data):]  # SMOTE-generated samples only
        
        available = synth.shape[0]
        step = min(step, available)
        
        generated[start:start + step] = synth[:step]
        start += step
        remaining -= step
        
        if remaining <= 0 or available == 0:
            break
            
    return generated


# ============================================================
#  3. NOISE + UTILITY LAYERS
# ============================================================

class GaussianNoise(nn.Module):
    """
    Applies multiplicative Gaussian noise during training.
    """

    def __init__(self, sigma: float = 0.1, detach_scale: bool = True):
        super().__init__()
        self.sigma = sigma
        self.detach_scale = detach_scale

    def forward(self, x: torch.Tensor):
        if not self.training or self.sigma <= 0:
            return x
        scale = x.detach() if self.detach_scale else x
        return x + torch.randn_like(x) * (self.sigma * scale)


class BiasLayer(nn.Module):
    """
    Adds a learnable bias vector.
    """

    def __init__(self, size: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return x + self.bias


class EpsReLU(nn.Module):
    """
    ReLU with a minimum epsilon instead of strict zero.
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.maximum(x, torch.full_like(x, self.eps))


# ============================================================
#  4. DIFFERENTIABLE STATISTIC LOSS FUNCTIONS
# ============================================================

def soft_count_nonzero(x, threshold: float = 3e-3, sharpness: float = 1000.0):
    """
    Differentiable approximation to counting non-zero pixels.
    """
    return torch.sigmoid(sharpness * (x - threshold)).sum(dim=(1, 2, 3))


def soft_threshold(x, threshold: float = 1e-3, sharpness: float = 1000.0):
    """
    Smooth threshold: output = x * sigmoid(k*(x - thresh))
    """
    k = sharpness
    return x * torch.sigmoid(k * (x - threshold))


def kde_kl_divergence_torch(
    real: torch.Tensor,
    fake: torch.Tensor,
    bandwidth: float = 0.1,
    num_points: int = 1000,
    eps: float = 1e-8
):
    """
    Compute KL divergence between two 1D distributions
    using differentiable Gaussian KDE approximations.
    """
    lo = min(real.min(), fake.min()).item()
    hi = max(real.max(), fake.max()).item()
    support = torch.linspace(lo, hi, num_points, device=real.device)

    def kde(samples):
        samples = samples.view(-1, 1)
        kernels = torch.exp(-0.5 * (samples - support) ** 2 / bandwidth**2)
        pdf = kernels.sum(dim=0)
        pdf /= (pdf.sum() + eps)
        return pdf + eps

    p = kde(real)
    q = kde(fake)
    return (p * (p.log() - q.log())).sum()
