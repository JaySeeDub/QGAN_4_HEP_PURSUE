"""
Data preparation utilities.
Includes JetDataset, signal filtering, KDE/SMOTE sampler creation,
and dataset packaging for training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset

from HEPGAN.helper import smotenc_generate, compute_kde_fits


# =========================================================
# ------------ Split dataset by signal/background ---------
# =========================================================
def signal_split(dataset: Dataset, train_on: str) -> Dataset:
    """
    Returns a filtered dataset:
        train_on = {"signal", "background", "both"}
    """
    if train_on == "both":
        return dataset

    sig_mask = dataset.features[:, 0] == 1
    bkg_mask = ~sig_mask

    if train_on == "signal":
        mask = sig_mask
    elif train_on == "background":
        mask = bkg_mask
    else:
        raise ValueError(f"Invalid train_on = {train_on}")

    indices = np.where(mask.cpu().numpy())[0]
    return Subset(dataset, indices).dataset


# =========================================================
# ------------------- Dataset Factory ----------------------
# =========================================================
def DataMaker(jet_mass_data, batch_size=128):
    """
    Construct:
        • dataset_signal
        • dataset_background
        • dataset_both

        plus SMOTE + KDE samplers for each variant.

    Returns:
        dict() of:
            "dataset_signal"
            "dataset_background"
            "dataset_both"
            "smoteSampler_signal"
            "smoteSampler_background"
            "smoteSampler_both"
            "kdeSampler_signal"
            "kdeSampler_background"
            "kdeSampler_both"
    """
    ######################
    # Just while debugging
    n_events = int(.01 * len(jet_mass_data["image"]))
    ######################
    
    # n_events = len(jet_mass_data["image"])
    datasets = {}

    base_dataset = JetDataset(jet_mass_data, n_events, Verbose=False)

    # Small dataset used only for SMOTE fitting
    smote_base = JetDataset(jet_mass_data, max(2000, int(0.01 * n_events)))

    train_modes = ["signal", "background", "both"]

    for mode in train_modes:

        # Filter for signal, background, both
        ds = signal_split(base_dataset, mode)
        datasets[f"dataset_{mode}"] = ds

        # ----- Create SMOTE sampler -----
        real_features = smote_base.features.numpy()
        smote_out = smotenc_generate(
            real_features,
            n_samples=len(base_dataset),
            categorical_features=[0]
        )
        datasets[f"smoteSampler_{mode}"] = torch.tensor(smote_out, dtype=torch.float32)

        # ----- KDE sampler -----
        datasets[f"kdeSampler_{mode}"] = compute_kde_fits(ds)

    return datasets


# =========================================================
# ---------------- JetDataset Definition -------------------
# =========================================================
class JetDataset(Dataset):
    """
    Prepares jet images + physics features.
    Computes:
        • normalized images
        • η-flipped images
        • ΔR mean/std
        • pixel mean/std
        • normalized mass / pt
    """

    def __init__(self, data, n_events, Verbose=False):
        # Crop: [6:-3, 5:-4]
        images = torch.tensor(
            data["image"][:n_events, 6:-3, 5:-4],
            dtype=torch.float32
        )

        # Normalize images to 0–1
        img_min, img_max = images.min(), images.max()
        images = (images - img_min) / (img_max - img_min)

        flipped = torch.flip(images, dims=[1])

        self.images = images
        self.flipped_images = flipped

        # =====================================================
        # ΔR and pixel statistics
        # =====================================================
        H, W = images[0].shape
        cx, cy = (W - 1) / 2, (H - 1) / 2

        x, y = torch.meshgrid(
            torch.arange(W, dtype=torch.float32),
            torch.arange(H, dtype=torch.float32),
            indexing="ij"
        )

        dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        weights = images
        dR = (weights * dist)

        dR_mean = dR.mean(dim=(1, 2))
        dR_std = dR.std(dim=(1, 2))
        pixel_mean = images.mean(dim=(1, 2))
        pixel_std = images.std(dim=(1, 2))

        # =====================================================
        # Feature tensor (true + derived)
        # =====================================================
        feats = np.stack([
            data["signal"][:n_events],
            data["jet_eta"][:n_events],
            data["jet_pt"][:n_events],
            data["jet_mass"][:n_events],
            data["jet_delta_R"][:n_events],
            dR_mean.numpy(),
            dR_std.numpy(),
            pixel_mean.numpy(),
            pixel_std.numpy()
        ], axis=1)

        feats = torch.tensor(feats, dtype=torch.float32)

        # Normalize pt & mass
        for idx in [2, 3]:
            v = feats[:, idx]
            feats[:, idx] = (v - v.min()) / (v.max() - v.min())

        # Flipped features (η → −η)
        flipped_feats = feats.clone()
        flipped_feats[:, 1] = -flipped_feats[:, 1]

        self.features = feats
        self.flipped_features = flipped_feats

        if Verbose:
            print(f"[JetDataset] Loaded {n_events} events")
            print(f"Image shape: {self.images.shape}")

    # =========================================================
    def __len__(self):
        return len(self.features)

    # =========================================================
    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.features[idx],
            self.flipped_images[idx],
            self.flipped_features[idx],
        )
