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

