# train.py
"""
Training loop for class-conditional GANs with two sampling backends:
 - KDE sampler (passed as dict or custom object)
 - SMOTE sampler (passed as DataLoader)
"""

from __future__ import annotations

import os
import math
import time
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from HEPGAN.diagnostics import (
    compute_distance_map,
    compute_fake_statistics,
    plot_generated_samples,
)
from HEPGAN.helper import kde_kl_divergence_torch, soft_count_nonzero, sample_from_kde
from HEPGAN.loadSaveEval import save

# For saving checkpoints
CHECKPOINT_DIR = "models"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def _safe_to_cpu(x):
    """
    Safely moves input to CPU without creating new tensors from existing tensors.
    Handles:
        - torch.Tensor
        - numpy arrays
        - Python scalars
        - lists, tuples, dicts (recursively)
    Ensures tensors are detached and moved to CPU.
    """
    import torch
    import numpy as np

    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).detach().cpu()

    if isinstance(x, (int, float)):
        return x  # no conversion needed

    if isinstance(x, list):
        return [_safe_to_cpu(v) for v in x]

    if isinstance(x, tuple):
        return tuple(_safe_to_cpu(v) for v in x)

    if isinstance(x, dict):
        return {k: _safe_to_cpu(v) for k, v in x.items()}

    raise TypeError(f"Unsupported type for _safe_to_cpu: {type(x)}")


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to device preserving dtype."""
    if t.device != device:
        return t.to(device)
    return t


def _ensure_batch_size_from_real(real_img: torch.Tensor, requested_bs: int) -> int:
    """
    Determine batch size to use for generated samples.
    Uses the number of real images in current batch (real_img.shape[0]).
    """
    bs = real_img.shape[0]
    # If dataloader returned fewer than requested, use that
    return bs if bs > 0 else requested_bs


def _sample_z_codings(
    sampleloader: Union[dict, torch.utils.data.DataLoader, torch.Tensor],
    batch_size: int,
    device: torch.device,
    num_coding_features: int = 9,
) -> torch.Tensor:
    """
    Return z_codings shaped (batch_size, num_coding_features) on `device`.

    Behavior:
      - If sampleloader is a dict (KDE style), we assume `sample_from_kde(sampleloader, ...)` works.
      - If sampleloader is a DataLoader, use next(iter(sampleloader)) to fetch a batch of codings.
      - If sampleloader is a torch.Tensor (precomputed tensor), sample rows randomly.
    """
    if isinstance(sampleloader, dict):
        # KDE: sample via your helper function
        z1 = torch.randint(0, 2, (batch_size, 1), device=device)
        z2 = sample_from_kde(sampleloader, num_samples=batch_size)[:, : (num_coding_features - 1)]
        if not isinstance(z2, torch.Tensor):
            z2 = torch.tensor(z2, dtype=torch.float32)
        z2 = z2.to(device=device, dtype=torch.float32)
        z_codings = torch.cat([z1, z2], dim=1).to(device)
    elif isinstance(sampleloader, torch.utils.data.DataLoader):
        # SMOTE DataLoader case: next(iter(...)) may return < batch_size
        try:
            z_batch = next(iter(sampleloader))
        except Exception:
            # If sampler is empty or exhausted, create random fallback
            z_batch = torch.rand((batch_size, num_coding_features))
        if isinstance(z_batch, (list, tuple)):
            # DataLoader may return (features, labels) etc.
            z_batch = z_batch[0]
        z_batch = z_batch.to(device=device, dtype=torch.float32)
        if z_batch.shape[0] < batch_size:
            # Repeat with replacement to reach requested batch size
            reps = math.ceil(batch_size / z_batch.shape[0])
            z_batch = z_batch.repeat(reps, 1)[:batch_size, :]
        z_codings = z_batch[:batch_size, :num_coding_features].to(device)
    elif isinstance(sampleloader, torch.Tensor):
        # precomputed tensor (e.g., smoteSampler stored as torch.tensor)
        idx = torch.randint(0, sampleloader.shape[0], (batch_size,))
        z_codings = sampleloader[idx].to(device=device, dtype=torch.float32)
    else:
        # fallback: random codings
        z_codings = torch.cat(
            [torch.randint(0, 2, (batch_size, 1), device=device),
             torch.randn((batch_size, num_coding_features - 1), device=device)],
            dim=1,
        ).to(device)
    return z_codings


def _safe_bce(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes BCE with correct dtype and device handling.
    preds: (B, 1) or (B,) float
    labels: same shape as preds
    """
    preds = preds.view(-1).float()
    labels = labels.view_as(preds).float()
    return nn.BCELoss()(preds, labels)


def train(
    model: Tuple[torch.nn.Module, torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    sampleloader: Union[dict, torch.utils.data.DataLoader, torch.Tensor],
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
    save_every: int = 10,
    debug: bool = False,
) -> Dict[str, list]:
    """
    Unified training loop for the GAN. Handles either KDE (dict) or SMOTE (DataLoader/Tensor).

    Args:
        model: (generator, discriminator)
        dataloader: yields (real_image, real_features, flipped_image, flipped_features)
        sampleloader: KDE dict OR DataLoader OR torch.Tensor (SMOTE codings)
        n_epochs, lr, batch_size: training hyperparams
        device: torch.device or None to auto-select
        save_every: checkpoint frequency (epochs)
        debug: if True, prints per-batch and per-epoch debug info

    Returns:
        dict containing tracked histories:
            { "g_losses", "d_losses",
              "validity_losses", "stat_losses", "nnz_losses",
              "loss_dR_mean", "loss_dR_std", "loss_pix_mean", "loss_pix_std" }
    """

    # ---------------------------------------------------------------------
    # Helpers (assume _sample_z_codings and _safe_bce are defined in module)
    # ---------------------------------------------------------------------
    def _debug_print(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    _debug_print("Device:", device)

    generator, discriminator = model
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))

    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=n_epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=n_epochs, eta_min=1e-6)

    # histories
    g_losses: list[float] = []
    d_losses: list[float] = []

    validity_hist: list[float] = []
    stat_hist: list[float] = []
    nnz_hist: list[float] = []

    loss_dR_mean_hist: list[float] = []
    loss_dR_std_hist: list[float] = []
    loss_pix_mean_hist: list[float] = []
    loss_pix_std_hist: list[float] = []

    stats_dict: Dict[str, list] = {
        "fake_dR_mean": [],
        "fake_dR_std": [],
        "fake_pixel_mean": [],
        "fake_pixel_std": [],
        "real_dR_mean": [],
        "real_dR_std": [],
        "real_pixel_mean": [],
        "real_pixel_std": [],
    }

    dists = compute_distance_map(16, 16).to(device)
    num_disc_coding = 4  # discriminator uses first 4 coding features

    # local helper to sample z for a given bs
    def _get_z_for_bs(bs: int) -> torch.Tensor:
        return _sample_z_codings(sampleloader, bs, device, num_coding_features=9)

    print("Starting training on device:", device)

    for epoch in range(n_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_validity = 0.0
        epoch_stat = 0.0
        epoch_nnz = 0.0

        epoch_loss_dR_mean = 0.0
        epoch_loss_dR_std = 0.0
        epoch_loss_pix_mean = 0.0
        epoch_loss_pix_std = 0.0

        n_batches = 0

        for i, batch in enumerate(dataloader):
            # expect (image, features, flipped_image, flipped_features)
            try:
                real_image, real_features, flipped_image, flipped_features = batch
            except Exception as exc:
                raise ValueError("Expected dataloader to yield (real_image, real_features, flipped_image, flipped_features)") from exc

            # move to device
            real_feat = real_features.to(device=device, dtype=torch.float32)
            real_flipped_feat = flipped_features.to(device=device, dtype=torch.float32)
            real_img = real_image.unsqueeze(1).to(device=device, dtype=torch.float32)
            real_flipped_img = flipped_image.unsqueeze(1).to(device=device, dtype=torch.float32)

            bs = real_img.shape[0]  # current batch size (handles partial last batch)

            # sample codings (shape [bs, 9])
            z_codings = _get_z_for_bs(bs)

            flipped_z_codings = z_codings.clone()
            if flipped_z_codings.shape[1] > 1:
                flipped_z_codings[:, 1] = -flipped_z_codings[:, 1]

            # --------------------
            # Discriminator step
            # --------------------
            if i % 3 == 0:
                discriminator.train()
                optimizer_D.zero_grad()

                fake_img = generator(z_codings)
                fake_flipped_img = generator(flipped_z_codings)

                real_disc_codings = real_feat[:, :num_disc_coding]
                real_flipped_disc_codings = real_flipped_feat[:, :num_disc_coding]
                fake_disc_codings = z_codings[:, :num_disc_coding]
                fake_flipped_disc_codings = flipped_z_codings[:, :num_disc_coding]

                real_pred = discriminator(real_img, real_disc_codings).view(-1)
                real_flipped_pred = discriminator(real_flipped_img, real_flipped_disc_codings).view(-1)
                fake_pred = discriminator(fake_img.detach(), fake_disc_codings).view(-1)
                fake_flipped_pred = discriminator(fake_flipped_img.detach(), fake_flipped_disc_codings).view(-1)

                preds_concat = torch.cat([real_pred, real_flipped_pred, fake_pred, fake_flipped_pred], dim=0)
                ones = torch.ones(real_pred.shape[0] + real_flipped_pred.shape[0], device=device)
                zeros = torch.zeros(fake_pred.shape[0] + fake_flipped_pred.shape[0], device=device)
                labels = torch.cat([ones, zeros], dim=0)

                d_loss = _safe_bce(preds_concat, labels)
                d_loss.backward()
                optimizer_D.step()
            else:
                d_loss = torch.tensor(0.0, device=device)

            # --------------------
            # Generator step
            # --------------------
            generator.train()
            optimizer_G.zero_grad()

            fake_img = generator(z_codings)
            fake_flipped_img = generator(flipped_z_codings)

            fake_disc_codings = z_codings[:, :num_disc_coding]
            fake_flipped_disc_codings = flipped_z_codings[:, :num_disc_coding]

            fake_pred = discriminator(fake_img, fake_disc_codings).view(-1)
            fake_flipped_pred = discriminator(fake_flipped_img, fake_flipped_disc_codings).view(-1)

            target_ones = torch.ones_like(fake_pred, device=device)
            validity_loss = _safe_bce(fake_pred, target_ones) + _safe_bce(fake_flipped_pred, target_ones)

            # --------------------
            # STATISTICS LOSS (componentized)
            # --------------------
            fake_stats_orig = compute_fake_statistics(fake_img.detach().to("cpu"), dists.detach().to("cpu"))
            fake_stats_flip = compute_fake_statistics(fake_flipped_img.detach().to("cpu"), dists.detach().to("cpu"))

            fake_dR_mean = torch.as_tensor(0.5 * (fake_stats_orig["fake_dR_mean"] + fake_stats_flip["fake_dR_mean"]),
                                           dtype=torch.float32, device=device)
            fake_dR_std = torch.as_tensor(0.5 * (fake_stats_orig["fake_dR_std"] + fake_stats_flip["fake_dR_std"]),
                                          dtype=torch.float32, device=device)
            fake_pixel_mean = torch.as_tensor(0.5 * (fake_stats_orig["fake_pixel_mean"] + fake_stats_flip["fake_pixel_mean"]),
                                              dtype=torch.float32, device=device)
            fake_pixel_std = torch.as_tensor(0.5 * (fake_stats_orig["fake_pixel_std"] + fake_stats_flip["fake_pixel_std"]),
                                             dtype=torch.float32, device=device)

            # real stats from z_codings
            real_dR_mean = z_codings[:, 5].to(device=device, dtype=torch.float32)
            real_dR_std = z_codings[:, 6].to(device=device, dtype=torch.float32)
            real_pixel_mean = z_codings[:, 7].to(device=device, dtype=torch.float32)
            real_pixel_std = z_codings[:, 8].to(device=device, dtype=torch.float32)

            # individual components (use L1Loss)
            loss_dR_mean = nn.L1Loss()(real_dR_mean, fake_dR_mean) / 0.8
            loss_dR_std = nn.L1Loss()(real_dR_std, fake_dR_std) / 0.025
            loss_pix_mean = nn.L1Loss()(real_pixel_mean, fake_pixel_mean) / 0.6
            loss_pix_std = nn.L1Loss()(real_pixel_std, fake_pixel_std) / 0.03

            stat_loss = (loss_dR_mean + loss_dR_std + loss_pix_mean + loss_pix_std) / 4.0

            # --------------------
            # NNZ loss
            # --------------------
            fake_nnz = soft_count_nonzero(fake_img, threshold=3e-3, sharpness=10000.0).to(device)
            real_nnz = soft_count_nonzero(real_img, threshold=3e-3, sharpness=10000.0).to(device)

            # make sure lengths align (broadcast to bs)
            fake_nnz = fake_nnz.view(-1)[:bs]
            real_nnz = real_nnz.view(-1)[:bs]

            nnz_loss = nn.MSELoss()(fake_nnz, real_nnz)

            # --------------------
            # Combine losses and step
            # --------------------
            alpha, beta, chi = 0.45, 0.0035, 0.03
            g_loss = alpha * validity_loss + beta * stat_loss + chi * nnz_loss

            g_loss.backward()
            optimizer_G.step()

            # --------------------
            # Accumulate per-batch numbers (for epoch averaging)
            # --------------------
            epoch_g_loss += float(g_loss.detach().cpu().item())
            epoch_d_loss += float(d_loss.detach().cpu().item())
            epoch_validity += float(validity_loss.detach().cpu().item())
            epoch_stat += float(stat_loss.detach().cpu().item())
            epoch_nnz += float(nnz_loss.detach().cpu().item())

            epoch_loss_dR_mean += float(loss_dR_mean.detach().cpu().item())
            epoch_loss_dR_std += float(loss_dR_std.detach().cpu().item())
            epoch_loss_pix_mean += float(loss_pix_mean.detach().cpu().item())
            epoch_loss_pix_std += float(loss_pix_std.detach().cpu().item())

            # optionally collect some stats (first few samples)
            stats_dict["fake_dR_mean"].append(_safe_to_cpu(fake_dR_mean[:min(8, fake_dR_mean.shape[0])]))
            stats_dict["fake_dR_std"].append(_safe_to_cpu(fake_dR_std[:min(8, fake_dR_std.shape[0])]))
            stats_dict["fake_pixel_mean"].append(_safe_to_cpu(fake_pixel_mean[:min(8, fake_pixel_mean.shape[0])]))
            stats_dict["fake_pixel_std"].append(_safe_to_cpu(fake_pixel_std[:min(8, fake_pixel_std.shape[0])]))
            stats_dict["real_dR_mean"].append(_safe_to_cpu(real_dR_mean[:min(8, real_dR_mean.shape[0])]))
            stats_dict["real_dR_std"].append(_safe_to_cpu(real_dR_std[:min(8, real_dR_std.shape[0])]))
            stats_dict["real_pixel_mean"].append(_safe_to_cpu(real_pixel_mean[:min(8, real_pixel_mean.shape[0])]))
            stats_dict["real_pixel_std"].append(_safe_to_cpu(real_pixel_std[:min(8, real_pixel_std.shape[0])]))

            n_batches += 1

        # --- end batch loop ---

        # scheduler step
        scheduler_G.step()
        scheduler_D.step()

        if n_batches == 0:
            _debug_print("No batches in epoch; skipping stats accumulation for this epoch.")
            continue

        # compute epoch averages
        avg_g = epoch_g_loss / n_batches
        avg_d = epoch_d_loss / n_batches
        avg_validity = epoch_validity / n_batches
        avg_stat = epoch_stat / n_batches
        avg_nnz = epoch_nnz / n_batches

        avg_loss_dR_mean = epoch_loss_dR_mean / n_batches
        avg_loss_dR_std = epoch_loss_dR_std / n_batches
        avg_loss_pix_mean = epoch_loss_pix_mean / n_batches
        avg_loss_pix_std = epoch_loss_pix_std / n_batches

        # append to histories
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        validity_hist.append(avg_validity)
        stat_hist.append(avg_stat)
        nnz_hist.append(avg_nnz)

        loss_dR_mean_hist.append(avg_loss_dR_mean)
        loss_dR_std_hist.append(avg_loss_dR_std)
        loss_pix_mean_hist.append(avg_loss_pix_mean)
        loss_pix_std_hist.append(avg_loss_pix_std)

        print(f"[Epoch {epoch+1}/{n_epochs}] [D loss: {avg_d:.4f}] [G loss: {avg_g:.4f}]")

        # Visualize some samples
        plot_generated_samples(generator, sampleloader, batch_size=min(16, batch_size))

        # detailed epoch debug
        _debug_print(
            "\n--- EPOCH STAT BREAKDOWN ---"
            f"\nAvg validity_loss  : {avg_validity:.6f}"
            f"\nAvg stat_loss      : {avg_stat:.6f}"
            f"\n  loss_dR_mean     : {avg_loss_dR_mean:.6f}"
            f"\n  loss_dR_std      : {avg_loss_dR_std:.6f}"
            f"\n  loss_pix_mean    : {avg_loss_pix_mean:.6f}"
            f"\n  loss_pix_std     : {avg_loss_pix_std:.6f}"
            f"\nAvg nnz_loss       : {avg_nnz:.6f}"

            "\n------------------------------\n"

            f"Epoch {epoch+1}: g_loss={g_loss.item():.6f}, d_loss={d_loss.item():.6f}"
            f"  validity_loss: {alpha * validity_loss.item()}", 
            f"  stat_loss: {be * stat_loss.item()}"
            f"  [dR_mean={loss_dR_mean.item():.6f}, dR_std={loss_dR_std.item():.6f}, pix_mean={loss_pix_mean.item():.6f}, pix_std={loss_pix_std.item():.6f}]"
            f"  nnz_loss: {chi * nnz_loss.item()}")
            
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"class_gan_epoch_{epoch+1}.pt")
            model = (generator, discriminator)
            save(model, save_path)
