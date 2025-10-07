#!/usr/bin/env python
# coding: utf-8

# Plotting
from Imports import *
from Helper import *
from Plotting import *

def plot_generated_samples_basic(generator, dataset, batch_size=16, latent_dim=256):

    # Generate fake data
    # Should be very easy to modify which values are passed as codings
    z_codings = torch.rand(batch_size, 256, 1, 1).to('cuda')
    z_codings = z_codings * 0.5 + 1
    
    vmin = dataset.images.min()
    vmax = dataset.images.max()

    with torch.no_grad():
        gen_samples = generator(z_codings)

    gen_samples = gen_samples.to('cpu')

    fig, axes = plt.subplots(1, min(batch_size, 16), figsize=(min(batch_size, 16), 1))
    for i in range(min(batch_size, 16)):
        axes[i].imshow(gen_samples[i, 0].numpy(), cmap= 'viridis', vmin=vmin, vmax=vmax)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def test_generated_samples_basic(
    generator,
    discriminator,
    dataset,
    batch_size=16,
    latent_dim=256,
    codings=None,
    plot_distributions=True,
    compare_discriminator=True
):

    # Generate fake data
    # Should be very easy to modify which values are passed as codings
    z_codings = torch.rand(batch_size, 256, 1, 1).to("cuda")
    z_codings = z_codings * 0.5 + 1

    with torch.no_grad():
        gen_samples = generator(z_codings)

    gen_samples = gen_samples.to('cpu')
    
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

        stats = track_statistics_basic(fake_stats)
        plot_tracked_statistics_basic(stats)

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
            pred1 = discriminator(real_imgs.unsqueeze(1).to('cuda')).detach()
            pred2 = discriminator(real_imgs.unsqueeze(1).to('cuda')).detach()

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
            real_preds = discriminator(real_input_imgs).detach().to('cpu').numpy().squeeze()
            fake_preds = discriminator(fake_input_imgs).detach().to('cpu').numpy().squeeze()

        # Ground truth: 1 for real, 0 for fake
        y_true = np.concatenate([np.ones_like(real_preds), np.zeros_like(fake_preds)])
        y_pred = np.concatenate([real_preds >= 0.5, fake_preds >= 0.5]).astype(int)
            
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, ["Fake", "Real"], "Real vs Generated", vmin=0, vmax=n_events)
        plot_confusion_matrix_percent(cm, ["Fake", "Real"], "Real vs Generated Samples")

def track_statistics_basic(fake_stats):
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

    return stats_dict

def plot_tracked_statistics_basic(stats_dict):

    fake_stats = [np.concatenate(stats_dict[f'fake_{k}']) for k in ['dR_mean', 'dR_std', 'pixel_mean', 'pixel_std']]

    stat_titles = ['ΔR Mean', 'ΔR Std', 'Pixel Mean', 'Pixel Std']

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    for i in range(4):
        ax = axs[i]
        fake_vals = fake_stats[i]

        lower = np.percentile(fake_vals, 1)
        upper = np.percentile(fake_vals, 99)

        fake_vals_trunc = fake_vals[(fake_vals >= lower) & (fake_vals <= upper)]

        n_samples = len(fake_vals_trunc)
        bins = max(10, int(np.sqrt(n_samples)/2))

        ax.hist(fake_vals_trunc, bins=bins, alpha=0.6, label='Fake', edgecolor='black', density=True, histtype='stepfilled')
        ax.set_xlim(lower, upper)
        ax.set_title(stat_titles[i])
        ax.legend()

    plt.tight_layout()
    plt.suptitle("Generated Distributions by Statistic", fontsize=16, y=1.02)
    plt.show()
