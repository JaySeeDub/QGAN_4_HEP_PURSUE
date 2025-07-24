#!/usr/bin/env python
# coding: utf-8

# Plotting
from Imports import *
from Helper import *

def plot_generated_samples(generator, dataset, kdes, batch_size=16, latent_dim=256):
      # Set to eval mode to disable dropout/batchnorm updates

    # Latent vectors
    # Should be very easy to modify which values are passed as codings
    z_codings = torch.cat([torch.randint(0, 2, (batch_size, 1)), 
                          sample_fit_noise(kdes, num_samples=batch_size)[:,:]],
                          dim=1).to('cuda')
    # z_noise = torch.randn(batch_size, 5, ).to('cuda')
    # z_feat = torch.cat([z_codings, z_noise], dim=1)
    z_feat = z_codings

    vmin = dataset.images.min()
    vmax = dataset.images.max()

    with torch.no_grad():
        gen_samples = generator(z_feat)

    gen_samples = gen_samples.to('cpu')
    
    print("Sample feature coding:", z_codings[1].to('cpu').numpy())

    fig, axes = plt.subplots(1, min(batch_size, 16), figsize=(min(batch_size, 16), 1))
    for i in range(min(batch_size, 16)):
        axes[i].imshow(gen_samples[i, 0].numpy(), cmap= 'viridis', vmin=vmin, vmax=vmax)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    
      # Restore training mode

def plot_real_samples(dataset):
    images = dataset.images
    N = images.shape[0]
    vmin = images.min()
    vmax = images.max()
    
    # Extract labels
    y_labels = dataset.features[:, 0].long()
    
    # Separate images by label
    images_0 = images[y_labels == 0]
    images_1 = images[y_labels == 1]
    
    # Compute mean images
    mean_0 = images_0.mean(dim=0)
    mean_1 = images_1.mean(dim=0)
    
    # --- Plot Mean Images (Separate Overlay Plots) ---
    fig1, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    im0 = axs[0].imshow(mean_0, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title("Mean Image: W Boson (Label 0)")
    axs[0].axis('off')
    fig1.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    im1 = axs[1].imshow(mean_1, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title("Mean Image: Background QCD (Label 1)")
    axs[1].axis('off')
    fig1.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    # --- Plot Grid of Random Images with Shared Colorbar ---
    n_rows, n_cols = 4, 16
    n_images = n_rows * n_cols
    
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows * 1.5))
    
    for i in range(n_images):
        n = torch.randint(N, (1,)).item()
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        im = ax.imshow(images[n], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
    
    # Shared colorbar
    fig2.subplots_adjust(right=0.9)
    cbar_ax = fig2.add_axes([.95, 0.15, 0.01, 0.7])
    fig2.colorbar(im, cax=cbar_ax)
    
    plt.show()

def plot_metrics(g_losses, d_losses):
    epochs = range(1, len(g_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, g_losses, label='Generator Loss', color='blue')
    plt.plot(epochs, d_losses, label='Discriminator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training Losses Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(dataset):
    # Stack all features
    all_features = torch.stack([dataset[i][1] for i in range(len(dataset))])
    
    feature_labels = [
        r"$\eta$", r"Mass", r"$p_T$", r"$\Delta R$",
        r"$\langle \Delta R \rangle$", r"$\sigma_{\Delta R}$",
        r"$\langle Pixel \rangle$", r"$\sigma_{Pixel}$"
    ]
    
    num_features = all_features.shape[1]
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    for i in range(num_features):
        axs[i].hist(all_features[:, i+1].to('cpu').numpy(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axs[i].set_title(f"Feature {i}: {feature_labels[i]}")
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.suptitle("Distributions of Input Features", fontsize=16, y=1.03)
    plt.show()

def test_generated_samples(
    generator,
    discriminator,
    dataset,
    kdes,
    batch_size=16,
    latent_dim=256,
    codings=None,
    plot_distributions=True,
    compare_discriminator=True
):
    
    

    # Latent vectors
    z_codings = torch.cat([torch.randint(0, 2, (batch_size, 1)), 
                          sample_fit_noise(kdes, num_samples=batch_size)[:,:]],
                          dim=1).to('cuda')

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

def plot_confusion_matrix(cm, labels, title, vmin=0, vmax=10000):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, vmin=vmin, vmax=vmax, cbar_kws={"label": "Count"})
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrix_percent(cm, labels, title):
    cm_percent = cm / cm.sum() * 200  # Normalize to percentages
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=100, cbar_kws={"label": "Percentage (%)"})
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def compute_distance_map(H, W):
    center_x, center_y = (W - 1) / 2, (H - 1) / 2
    x_coords, y_coords = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing='ij'
    )
    dists = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    return dists.unsqueeze(0)

def compute_fake_statistics(fake_img, dists):
    weights = fake_img.squeeze(1).detach()
    stats = {
        'fake_dR_mean': (weights * dists).mean(dim=(1, 2)),
        'fake_dR_std': (weights * dists).std(dim=(1, 2)),
        'fake_pixel_mean': weights.mean(dim=(1, 2)),
        'fake_pixel_std': weights.std(dim=(1, 2)),
    }
    
    return stats

def track_statistics(fake_stats, real_stats):
    """
    Appends detached CPU copies of statistics to the tracking dictionary.
    """
    stats_dict = {
    'fake_dR_mean': [],
    'fake_dR_std': [],
    'fake_pixel_mean': [],
    'fake_pixel_std': [],
    'real_dR_mean': [],
    'real_dR_std': [],
    'real_pixel_mean': [],
    'real_pixel_std': []
}
    for key, val in fake_stats.items():
        stats_dict[f'{key}'].append(val.detach().to('cpu'))
    for key, val in real_stats.items():
        stats_dict[f'{key}'].append(val.detach().to('cpu'))

    return stats_dict

def plot_tracked_statistics(stats_dict):

    fake_stats = [np.concatenate(stats_dict[f'fake_{k}']) for k in ['dR_mean', 'dR_std', 'pixel_mean', 'pixel_std']]
    real_stats = [np.concatenate(stats_dict[f'real_{k}']) for k in ['dR_mean', 'dR_std', 'pixel_mean', 'pixel_std']]

    stat_titles = ['ΔR Mean', 'ΔR Std', 'Pixel Mean', 'Pixel Std']

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    for i in range(4):
        ax = axs[i]
        real_vals, fake_vals = real_stats[i], fake_stats[i]

        lower = min(np.percentile(real_vals, 1), np.percentile(fake_vals, 1))
        upper = max(np.percentile(real_vals, 99), np.percentile(fake_vals, 99))

        real_vals_trunc = real_vals[(real_vals >= lower) & (real_vals <= upper)]
        fake_vals_trunc = fake_vals[(fake_vals >= lower) & (fake_vals <= upper)]

        n_samples = min(len(real_vals_trunc), len(fake_vals_trunc))
        bins = max(10, int(np.sqrt(n_samples)/2))

        ax.hist(real_vals_trunc, bins=bins, alpha=0.6, label='Real', edgecolor='black', density=True, histtype='stepfilled')
        ax.hist(fake_vals_trunc, bins=bins, alpha=0.6, label='Fake', edgecolor='black', density=True, histtype='stepfilled')
        ax.set_xlim(lower, upper)
        ax.set_title(stat_titles[i])
        ax.legend()

    plt.tight_layout()
    plt.suptitle("Real vs Fake Distributions by Statistic", fontsize=16, y=1.02)
    plt.show()
