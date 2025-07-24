#!/usr/bin/env python
# coding: utf-8
from Imports import *

## Helper Functions for kde distributions, noise, loss, etc
def feature_distributions(dataset):
    all_features = torch.stack([dataset[i][1] for i in range(len(dataset))])
    
    feature_labels = [
        r"$\eta$", r"Mass", r"$p_T$", r"$\Delta R$",
        r"$\langle \Delta R \rangle$", r"$\sigma_{\Delta R}$",
        r"$\langle \mathrm{Pixel} \rangle$", r"$\sigma_{\mathrm{Pixel}}$"
    ]
    
    num_features = all_features.shape[1]
    fig, axs = plt.subplots(2, 4, figsize=(20, 6))
    axs = axs.flatten()

    kde_fits = {}
    
    for i in range(num_features-1):
        data = all_features[:, i+1].cpu().numpy()
        mean_val = data.mean()
        std_val = data.std()

        # Histogram
        counts, bins, _ = axs[i].hist(data, bins=50, alpha=0.4, color='skyblue', edgecolor='black', density=True)

        # KDE fit
        kde = gaussian_kde(data)
        kde_fits[feature_labels[i]] = kde
        x_vals = np.linspace(bins[0], bins[-1], 500)
        axs[i].plot(x_vals, kde(x_vals), label='KDE', color='green')
        axs[i].set_title(feature_labels[i], fontsize=12)
        axs[i].set_xlabel(feature_labels[i])
        axs[i].set_ylabel("Density")
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout()
    plt.suptitle("Feature Distributions with KDE Fits", fontsize=16, y=1.03)
    plt.show()
    return kde_fits

def sample_fit_noise(kde_fits, num_samples):
    
    feature_labels = [
        r"$\eta$", r"Mass", r"$p_T$", r"$\Delta R$",
        r"$\langle \Delta R \rangle$", r"$\sigma_{\Delta R}$",
        r"$\langle \mathrm{Pixel} \rangle$", r"$\sigma_{\mathrm{Pixel}}$"
    ]

    samples = []
    for label in feature_labels:
        kde = kde_fits[label]
        sampled = kde.resample(num_samples).T.squeeze()
        samples.append(sampled)

    stacked = np.stack(samples, axis=1)  # shape (num_samples, 9)
    return torch.tensor(stacked, dtype=torch.float32)

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        if self.training and self.sigma > 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = torch.randn_like(x) * scale
            return x + sampled_noise
        return x

def kde_kl_divergence_torch(real, fake, bandwidth=0.1, num_points=1000, eps=1e-8):
    min_val = torch.min(real.min(), fake.min()).detach()
    max_val = torch.max(real.max(), fake.max()).detach()
    support = torch.linspace(min_val, max_val, num_points, device=real.device).view(1, -1)  # [1, num_points]

    def kde(samples):
        # [B, 1] - expand to [B, num_points] for distance to each x
        samples = samples.view(-1, 1)
        dists = (samples - support) ** 2  # [B, num_points]
        kernels = torch.exp(-0.5 * dists / bandwidth**2)
        pdf = kernels.sum(dim=0)  # sum over samples -> [num_points]
        pdf /= (pdf.sum() + eps)  # normalize
        return pdf + eps  # avoid log(0)

    p = kde(real)
    q = kde(fake)

    kl = (p * (p.log() - q.log())).sum()
    return kl
    
def MaxReLU(x):
    return torch.minimum(x, torch.tensor(1))

def soft_count_nonzero(x, threshold=3e-3, sharpness=1000.0):
    """
    Soft count of non-zero pixels using a sigmoid approximation.

    Args:
        x (Tensor): shape [batch, 1, H, W]
        threshold (float): value below which pixels are 'effectively zero'
        sharpness (float): controls the steepness of the sigmoid

    Returns:
        Tensor: soft non-zero pixel counts, shape [batch]
    """
    return torch.sigmoid(sharpness * (x - threshold)).sum(dim=(1, 2, 3))

def soft_threshold(x, threshold=0.001, sharpness=1000.0):
    return x * torch.sigmoid(sharpness * (x - threshold))

class eps_relu(nn.Module):
    def __init__(self, epsilon):
        super(eps_relu, self).__init__()
        self.eps = epsilon
    def forward(self, x):
        return torch.maximum(x, torch.zeros_like(x) + self.eps)

class BiasLayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size))  # Learnable bias

    def forward(self, x):
        return x + self.bias
