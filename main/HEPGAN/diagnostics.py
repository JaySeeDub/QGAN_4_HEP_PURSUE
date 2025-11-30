"""
diagnostics.py — Utilities for plotting, visualization, and GAN diagnostics
"""

# ==========================================
# Imports
# ==========================================
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from HEPGAN.helper import sample_from_kde


# ==========================================
# Utility Functions
# ==========================================
def _device_of(model):
    """Return the device where a model's parameters live."""
    return next(model.parameters()).device


def _safe_to_cpu(x):
    return x.detach().to("cpu")


# ==========================================
# --- Basic Image Plotting ---
# ==========================================
def plot_generated_samples(generator, kdes, batch_size=16):
    """Generate and plot a row of generated samples conditioned on KDE noise."""
    device = _device_of(generator)

    # SMOTE DataLoader case: next(iter(...)) may return < batch_size
    try:
        z_feats = next(iter(sampleloader))
    except Exception:
           
        # Try to sample from KDE
        z_feats = torch.cat([
            torch.randint(0, 2, (batch_size, 1)),
            sample_from_kde(kdes, batch_size)
        ], dim=1).float().to(device)

    with torch.no_grad():
        fake = generator(z_feats)

    fake = _safe_to_cpu(fake)

    fig, axes = plt.subplots(1, min(batch_size, 16), figsize=(min(batch_size, 16), 1))
    for i in range(min(batch_size, 16)):
        axes[i].imshow(fake[i, 0], cmap='viridis')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def plot_real_samples(dataset):
    """Plot mean images and random real samples with shared colormap."""
    images = dataset.images
    N = images.shape[0]
    vmin, vmax = images.min(), images.max()
    labels = dataset.features[:, 0].long()

    img0 = images[labels == 0]
    img1 = images[labels == 1]
    mean0 = img0.mean(dim=0)
    mean1 = img1.mean(dim=0)

    # Mean images
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(mean0, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0].set_title("Mean W Boson")
    axs[0].axis("off")
    axs[1].imshow(mean1, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[1].set_title("Mean Background")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()

    # Random grid
    n_rows, n_cols = 4, 16
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows*1.5))
    for ax in axes.ravel():
        idx = torch.randint(0, N, (1,)).item()
        im = ax.imshow(images[idx], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis("off")

    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([.95, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cax)
    plt.show()


# ==========================================
# Training Loss Plotting
# ==========================================
def plot_metrics(g_losses, d_losses):
    """Plot generator/discriminator loss curves."""
    epochs = range(1, len(g_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, g_losses, label='Generator', color='blue')
    plt.plot(epochs, d_losses, label='Discriminator', color='red')
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================
# Feature Distribution Visualization
# ==========================================
def plot_feature_distributions(dataset):
    """Plot histograms of all physics-informed input features."""
    feats = torch.stack([dataset[i][1] for i in range(len(dataset))])
    names = [
        r"$\eta$", r"Mass", r"$p_T$", r"$\Delta R$",
        r"$\langle \Delta R \rangle$", r"$\sigma_{\Delta R}$",
        r"$\langle Pixel \rangle$", r"$\sigma_{Pixel}$"
    ]

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs = axs.flatten()

    for i in range(len(names)):
        axs[i].hist(_safe_to_cpu(feats[:, i+1]).numpy(), bins=50, color='skyblue', edgecolor='black')
        axs[i].set_title(names[i])
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


# ==========================================
# --- Distance-Based Statistics ---
# ==========================================
def compute_distance_map(H, W):
    """Radial distance from center for each pixel."""
    cx, cy = (W - 1) / 2, (H - 1) / 2
    x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij')
    d = torch.sqrt((x - cx)**2 + (y - cy)**2)
    return d[None]


def compute_fake_statistics(fake_imgs, dmap):
    """Compute ΔR mean/std and pixel mean/std for fake images."""
    w = fake_imgs.squeeze(1)
    return {
        "fake_dR_mean": (w * dmap).mean((1, 2)),
        "fake_dR_std": (w * dmap).std((1, 2)),
        "fake_pixel_mean": w.mean((1, 2)),
        "fake_pixel_std": w.std((1, 2)),
    }


def track_statistics(fake_stats, real_stats):
    """Track fake/real stat samples for histogram plotting."""
    keys = [
        "fake_dR_mean", "fake_dR_std", "fake_pixel_mean", "fake_pixel_std",
        "real_dR_mean", "real_dR_std", "real_pixel_mean", "real_pixel_std"
    ]
    return {k: [_safe_to_cpu(fake_stats[k] if "fake" in k else real_stats[k])]
            for k in keys}


def plot_tracked_statistics(stats):
    """Overlay histograms comparing fake vs real distributions."""
    fake_keys = ["dR_mean", "dR_std", "pixel_mean", "pixel_std"]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i, k in enumerate(fake_keys):
        ax = axs[i]
        real = np.concatenate(stats[f"real_{k}"])
        fake = np.concatenate(stats[f"fake_{k}"])

        # Robust bounds
        lo = min(np.percentile(real, 1), np.percentile(fake, 1))
        hi = max(np.percentile(real, 99), np.percentile(fake, 99))

        real_trim = real[(real >= lo) & (real <= hi)]
        fake_trim = fake[(fake >= lo) & (fake <= hi)]
        bins = max(10, int(np.sqrt(min(len(real_trim), len(fake_trim))) / 2))

        ax.hist(real_trim, bins=bins, alpha=0.6, label="Real", histtype="stepfilled")
        ax.hist(fake_trim, bins=bins, alpha=0.6, label="Fake", histtype="stepfilled")
        ax.set_title(k)
        ax.set_xlim(lo, hi)
        ax.legend()

    plt.tight_layout()
    plt.show()


# ==========================================
# --- Confusion Matrix Utilities ---
# ==========================================
def plot_confusion_matrix(cm, labels, title, vmin=0, vmax=None):
    """Plot raw confusion matrix."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                vmin=vmin, vmax=vmax, cbar_kws={"label": "Count"})
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_percent(cm, labels, title):
    """Plot percentage confusion matrix."""
    cm_pct = cm / cm.sum() * 100
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=100, cbar_kws={"label": "Percent"})
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


# ==========================================
# --- Full GAN Diagnostic Test ---
# ==========================================
def test_generated_samples(
        generator,
        discriminator,
        dataset,
        kdes,
        batch_size=16,
        plot_distributions=True,
        compare_discriminator=True
):
    """End-to-end diagnostic: generate samples, plot stats, test discriminator."""
    device = _device_of(generator)

    z = torch.cat([
        torch.randint(0, 2, (batch_size, 1)),
        sample_from_kde(kdes, batch_size)
    ], dim=1).float().to(device)

    with torch.no_grad():
        fake = generator(z)

    fake_cpu = _safe_to_cpu(fake)
    vmin, vmax = dataset.images.min(), dataset.images.max()

    # Show images
    fig, axes = plt.subplots(1, min(batch_size, 16), figsize=(16, 2))
    for i in range(min(batch_size, 16)):
        axes[i].imshow(fake_cpu[i, 0], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

    # Statistics
    if plot_distributions:
        dmap = compute_distance_map(16, 16)
        fake_stats = compute_fake_statistics(fake_cpu, dmap)

        real_stats = {
            "real_dR_mean": z[:, 5],
            "real_dR_std": z[:, 6],
            "real_pixel_mean": z[:, 7],
            "real_pixel_std": z[:, 8],
        }

        stats = track_statistics(fake_stats, real_stats)
        plot_tracked_statistics(stats)

    # Discriminator tests
    if compare_discriminator:
        _run_discriminator_diagnostic(discriminator, dataset, fake_cpu, z)


def test_smote_samples(
    generator,
    discriminator,
    dataset,
    minority,
    batch_size=16,
    latent_dim=256,
    codings=None,
    plot_distributions=True,
    compare_discriminator=True
):
    
    # Latent vectors
    z_codings = minority[:batch_size]
    z_codings = torch.tensor(z_codings, dtype=torch.float32).to(next(generator.parameters()).device)

    z_feat = z_codings

    with torch.no_grad():
        gen_samples = generator(z_codings)

    gen_samples = gen_samples.to('cpu')

    print("Sample feature coding:", z_codings[1].to('cpu').numpy())
    
    vmin = dataset.images.min()
    vmax = dataset.images.max()
    fig, axes = plt.subplots(1, min(batch_size, 16), figsize=(min(batch_size, 16), 1))
    
    for i in range(min(batch_size, 16)):
        n = torch.randint(batch_size, ())
        axes[i].imshow(gen_samples[n, 0].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    if plot_distributions:
        dists = compute_distance_map(16, 16)
        fake_stats = compute_fake_statistics(gen_samples, dists)

        # Get real stats from z_codings (features 5–8)
        real_dR_mean = z_codings[:,5]
        real_dR_std = z_codings[:,6]
        real_pixel_mean = z_codings[:,7]
        real_pixel_std = z_codings[:,8]
    
        real_stats = {
            'real_dR_mean': real_dR_mean,
            'real_dR_std': real_dR_std,
            'real_pixel_mean': real_pixel_mean,
            'real_pixel_std': real_pixel_std
        }
        stats = track_statistics(fake_stats, real_stats)
        plot_tracked_statistics(stats)

    if compare_discriminator:
        n_events = batch_size
        real_features = dataset.features[:int(n_events/2), :4].clone()
        real_features = torch.cat([real_features, dataset.features[int(-n_events/2):, :4].clone()], 0)
        real_imgs = dataset.images[:int(n_events/2)].clone()
        real_imgs = torch.cat([real_imgs, dataset.images[int(-n_events/2):].clone()], 0)
        real_labels = real_features[:, 0]

        test_features = real_features.clone()
        test_labels = torch.zeros_like(real_labels)
        test_features[:, 0] *= (-2**(test_features[:, 0])+2)
        with torch.no_grad():
            pred1 = discriminator(real_imgs.unsqueeze(1).to('cuda'), test_features.to('cuda')).detach()
            pred2 = discriminator(real_imgs.unsqueeze(1).to('cuda'), real_features.to('cuda')).detach()

        print("Discriminator output (wrong label):", pred1.squeeze().to('cpu').numpy())
        print("Discriminator output (correct label):", pred2.squeeze().to('cpu').numpy())
        print("Real labels:", real_labels.numpy())
        print("Swapped labels:", test_features[:, 0].numpy())
        print("Relative change (%):", ((pred1 / pred2 - 1) * 100).squeeze().to('cpu').numpy())

        # Confusion matrix for real data, wrong label
        # Threshold predictions
        # pred1 = discriminator(real_imgs, fake_labels) → expect 0
        # pred2 = discriminator(real_imgs, true_labels) → expect 1
        pred1_binary = (pred1.to('cpu').numpy() >= 0.5).astype(int).squeeze()
        pred2_binary = (pred2.to('cpu').numpy() >= 0.5).astype(int).squeeze()
        
        # Create true labels
        true_labels = np.concatenate([np.zeros_like(pred1_binary), np.ones_like(pred2_binary)])
        predicted_labels = np.concatenate([pred1_binary, pred2_binary])

        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        plot_confusion_matrix(cm, ["Fake", "Real"], "Signal vs Background", vmin=0, vmax=n_events)
        plot_confusion_matrix_percent(cm, ["Fake", "Real"], "Signal vs Background")

        # Confusion matrix for generated samples
        with torch.no_grad():
            fake_imgs = generator(z_codings)
            fake_feats = z_codings[:, :4]  # discriminator expects first 4 features

        # Make sure we use the same number of real and fake samples
        n_events = min(len(fake_feats), len(real_features))
        real_input_imgs = real_imgs[:n_events].unsqueeze(1).to('cuda')
        real_input_feats = real_features[:n_events].to('cuda')

        fake_input_imgs = fake_imgs[:n_events].to('cuda')
        fake_input_feats = fake_feats[:n_events].to('cuda')

        # Get predictions
        with torch.no_grad():
            real_preds = discriminator(real_input_imgs, real_input_feats).detach().to('cpu').numpy().squeeze()
            fake_preds = discriminator(fake_input_imgs, fake_input_feats).detach().to('cpu').numpy().squeeze()

        # Ground truth: 1 for real, 0 for fake
        y_true = np.concatenate([np.ones_like(real_preds), np.zeros_like(fake_preds)])
        y_pred = np.concatenate([real_preds >= 0.5, fake_preds >= 0.5]).astype(int)
            
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, ["Fake", "Real"], "Real vs Generated", vmin=0, vmax=n_events)
        plot_confusion_matrix_percent(cm, ["Fake", "Real"], "Real vs Generated Samples")
        

def _run_discriminator_diagnostic(discriminator, dataset, fake_imgs, fake_feats):
    device = _device_of(discriminator)

    N = fake_imgs.shape[0]
    real_imgs = dataset.images[:N].unsqueeze(1).to(device)
    real_feats = dataset.features[:N, :4].to(device)

    fake_imgs = fake_imgs[:N].to(device)
    fake_feats = fake_feats[:, :4].to(device)

    with torch.no_grad():
        real_pred = discriminator(real_imgs, real_feats).cpu().numpy().squeeze()
        fake_pred = discriminator(fake_imgs, fake_feats).cpu().numpy().squeeze()

    y_true = np.concatenate([np.ones_like(real_pred), np.zeros_like(fake_pred)])
    y_pred = np.concatenate([(real_pred >= 0.5), (fake_pred >= 0.5)]).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, ["Fake", "Real"], "Real vs Generated")
    plot_confusion_matrix_percent(cm, ["Fake", "Real"], "Real vs Generated (%)")
