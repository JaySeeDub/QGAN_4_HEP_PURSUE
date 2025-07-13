#!/usr/bin/env python
# coding: utf-8

# Plotting
from Imports import *
from Helper import *

def plot_generated_samples(generator, dataset, kdes, batch_size=16, latent_dim=256):
    generator.eval()  # Set to eval mode to disable dropout/batchnorm updates

    # Latent vectors
    z_img = torch.randn(batch_size, latent_dim, 1, 1).cuda()
    # Should be very easy to modify which values are passed as codings
    z_codings = torch.cat([torch.randint(0, 2, (batch_size, 1)), 
                          sample_fit_noise(kdes, num_samples=batch_size)[:,:]],
                          dim=1).cuda()
    # z_noise = torch.randn(batch_size, 5, ).cuda()
    # z_feat = torch.cat([z_codings, z_noise], dim=1)
    z_feat = z_codings

    vmin = dataset.images[:batch_size].min()
    vmax = dataset.images[:batch_size].max()

    with torch.no_grad():
        gen_samples = generator(z_feat)

    gen_samples = gen_samples.cpu()
    
    print("Sample feature coding:", z_codings[1].cpu().numpy())

    fig, axes = plt.subplots(1, min(batch_size, 16), figsize=(min(batch_size, 16), 1))
    for i in range(min(batch_size, 16)):
        axes[i].imshow(gen_samples[i, 0].numpy(), cmap= 'viridis', vmin=vmin, vmax=vmax)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    
    generator.train()  # Restore training mode

def plot_metrics(g_losses, d_losses):
    epochs = range(1, len(g_losses) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, g_losses, label='Generator Loss', color='blue')
    plt.plot(epochs, d_losses, label='Discriminator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
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
        axs[i].hist(all_features[:, i+1].cpu().numpy(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
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
    generator.eval()
    discriminator.eval()

    # Latent vectors
    z_img = torch.randn(batch_size, latent_dim, 1, 1).cuda()
    z_codings = torch.cat([
        torch.randint(0, 2, (batch_size, 1)),
        sample_fit_noise(kdes, num_samples=batch_size)[:, :]
    ], dim=1).cuda()

    with torch.no_grad():
        gen_samples = generator(z_codings)

    gen_samples = gen_samples.cpu()

    print("Sample feature coding:", z_codings[1].cpu().numpy())

    fig, axes = plt.subplots(1, min(batch_size, 16), figsize=(min(batch_size, 16), 1))
    for i in range(min(batch_size, 16)):
        axes[i].imshow(gen_samples[i, 0].numpy(), cmap='viridis')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    if plot_distributions:
        dists = compute_distance_map(16, 16).cuda()
        stats = compute_statistics(z_codings, gen_samples, dists)
        plot_stat_distributions(stats)

    if compare_discriminator:
        n_events = batch_size
        real_features = dataset.features[:n_events, :4].clone()
        real_features = torch.cat([real_features, dataset.features[-n_events:, :4].clone()], 0)
        real_imgs = dataset.images[:n_events].clone()
        real_imgs = torch.cat([real_imgs, dataset.images[-n_events:].clone()], 0)
        real_labels = real_features[:, 0]

        test_features = real_features.clone()
        test_labels = torch.zeros_like(real_labels)
        test_features[:, 0] *= (-2**(test_features[:, 0])+2)

        pred1 = discriminator(real_imgs.unsqueeze(1).cuda(), test_features.cuda()).detach()
        pred2 = discriminator(real_imgs.unsqueeze(1).cuda(), real_features.cuda()).detach()

        print("Discriminator output (wrong label):", pred1.squeeze().cpu().numpy())
        print("Discriminator output (correct label):", pred2.squeeze().cpu().numpy())
        print("Real labels:", real_labels.numpy())
        print("Swapped labels:", test_features[:, 0].numpy())
        print("Relative change (%):", ((pred1 / pred2 - 1) * 100).squeeze().cpu().numpy())

        # Confusion matrix for real data, wrong label
        # Threshold predictions
        # pred1 = discriminator(real_imgs, fake_labels) → expect 0
        # pred2 = discriminator(real_imgs, true_labels) → expect 1
        pred1_binary = (pred1.cpu().numpy() >= 0.5).astype(int).squeeze()
        pred2_binary = (pred2.cpu().numpy() >= 0.5).astype(int).squeeze()
        
        # Create true labels
        true_labels = np.concatenate([np.zeros_like(pred1_binary), np.ones_like(pred2_binary)])
        predicted_labels = np.concatenate([pred1_binary, pred2_binary])

        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Discriminator Confusion Matrix (Real Samples Only)")

        # Confusion matrix for generated samples
        with torch.no_grad():
            fake_imgs = generator(z_codings)
            fake_feats = z_codings[:, :4]  # discriminator expects first 4 features

        # Make sure we use the same number of real and fake samples
        n = min(len(fake_feats), len(real_features))
        real_input_imgs = real_imgs[:n].unsqueeze(1).cuda()
        real_input_feats = real_features[:n].cuda()
        
        fake_input_imgs = fake_imgs[:n].cuda()
        fake_input_feats = fake_feats[:n].cuda()
        
        # Get predictions
        real_preds = discriminator(real_input_imgs, real_input_feats).detach().cpu().numpy().squeeze()
        fake_preds = discriminator(fake_input_imgs, fake_input_feats).detach().cpu().numpy().squeeze()
        
        # Ground truth: 1 for real, 0 for fake
        y_true = np.concatenate([np.ones(n), np.zeros(n)])
        y_pred = np.concatenate([real_preds >= 0.5, fake_preds >= 0.5]).astype(int)
            
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    
        # Plot
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Discriminator Confusion Matrix (Real vs Generated Samples)")
        plt.show()

    generator.train()
    discriminator.train()


def compute_distance_map(H, W):
    center_x, center_y = (W - 1) / 2, (H - 1) / 2
    x_coords, y_coords = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing='ij'
    )
    dists = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    return dists.unsqueeze(0)

def compute_statistics(z_codings, fake_img, dists):
    weights = fake_img.squeeze(1).cuda()

    return {
        'fake_dR_mean': (weights * dists).mean(dim=(1, 2)).cpu().numpy(),
        'fake_dR_std': (weights * dists).std(dim=(1, 2)).cpu().numpy(),
        'fake_pixel_mean': weights.mean(dim=(1, 2)).cpu().numpy(),
        'fake_pixel_std': weights.std(dim=(1, 2)).cpu().numpy(),
        'real_dR_mean': z_codings[:, 5].cpu().numpy(),
        'real_dR_std': z_codings[:, 6].cpu().numpy(),
        'real_pixel_mean': z_codings[:, 7].cpu().numpy(),
        'real_pixel_std': z_codings[:, 8].cpu().numpy(),
    }


def plot_stat_distributions(stats):
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    stat_titles = ['ΔR Mean', 'ΔR Std', 'Pixel Mean', 'Pixel Std']

    for i, key in enumerate(['dR_mean', 'dR_std', 'pixel_mean', 'pixel_std']):
        ax = axs[i]
        real_vals = stats[f'real_{key}']
        fake_vals = stats[f'fake_{key}']

        lower = min(np.percentile(real_vals, 1), np.percentile(fake_vals, 1))
        upper = max(np.percentile(real_vals, 99), np.percentile(fake_vals, 99))

        real_vals_trunc = real_vals[(real_vals >= lower) & (real_vals <= upper)]
        fake_vals_trunc = fake_vals[(fake_vals >= lower) & (fake_vals <= upper)]

        ax.hist(real_vals_trunc, bins=1000, alpha=0.6, label='Real',
                edgecolor='black', density=True, histtype='stepfilled')
        ax.hist(fake_vals_trunc, bins=1000, alpha=0.6, label='Fake',
                edgecolor='black', density=True, histtype='stepfilled')

        ax.set_xlim(lower, upper)
        ax.set_title(stat_titles[i])
        ax.legend()

    plt.tight_layout()
    plt.suptitle("Real vs Fake Distributions by Statistic", fontsize=16, y=1.02)
    plt.show()