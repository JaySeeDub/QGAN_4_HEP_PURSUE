#!/usr/bin/env python
# coding: utf-8
from Imports import *

# Dataset Preprocessing
class JetDataset(Dataset):
    def __init__(self, data, n_events):
        self.images = torch.tensor(data['image'][:n_events, 6:-3, 5:-4], dtype=torch.float32)
        self.flipped_images = torch.flip(self.images,[1])

        # Normalize
        self.images = (self.images - self.images.min()) / (self.images.max() - self.images.min())
        self.flipped_images = (self.flipped_images - self.flipped_images.min()) / (self.flipped_images.max() - self.flipped_images.min())

        # ----- ΔR Calculation -----
        print(f"Image shape: {self.images.shape}")
        H, W = self.images[0].shape
        center_x, center_y = (W - 1) / 2, (H - 1) / 2  # center = (12, 12)

        # Coordinate grid
        x_coords, y_coords = torch.meshgrid(
            torch.arange(W, dtype=torch.float32),
            torch.arange(H, dtype=torch.float32),
            indexing='ij'
        )

        # Distance from center
        dists = torch.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

        # Weighted: sum(pixel * distance) / sum(pixel)
        weights = self.images
        weight_norm = self.images.max()

        dR = (weights * dists) / weight_norm

        # dim = (1,2,3)
        dR_mean = dR.mean(dim = (1,2))
        dR_std = dR.std(dim = (1,2))
        print(f"dR Mean: {dR_mean.shape}")
        print(f"dR STD: {dR_std.shape}")
        
        # Pixel stats
        pixel_mean = weights.mean(dim = (1,2))
        pixel_std = weights.std(dim = (1,2))
        print(f"Pixel Mean: {pixel_mean.shape}")
        print(f"Pixel STD: {pixel_std.shape}")

        self.features = torch.tensor(np.stack([
            data['signal'][:n_events],
            data['jet_eta'][:n_events],
            data['jet_pt'][:n_events],
            data['jet_mass'][:n_events],
            data['jet_delta_R'][:n_events],
            dR_mean, 
            dR_std, 
            pixel_mean, 
            pixel_std
        ], axis=1), dtype=torch.float32)
        
        self.flipped_features = torch.tensor(np.stack([
            data['signal'][:n_events],
            -data['jet_eta'][:n_events],
            data['jet_pt'][:n_events],
            data['jet_mass'][:n_events],
            data['jet_delta_R'][:n_events],
            dR_mean, 
            dR_std, 
            pixel_mean, 
            pixel_std,
        ], axis=1), dtype=torch.float32)

        # Normalize just pt and mass features here also
        # Normalize jet_mass (index 2)
        self.features[:, 2] = (self.features[:, 2]-self.features[:, 2].min()) / (self.features[:, 2].abs().max()-self.features[:, 2].min())
        
        # Normalize jet_pt (index 3)
        self.features[:, 3] = (self.features[:, 3]-self.features[:, 3].min()) / (self.features[:, 3].abs().max()-self.features[:, 3].min())
        
        # Same for flipped features
        self.flipped_features[:, 2] = (self.flipped_features[:, 2]-self.flipped_features[:, 2].min()) / (self.flipped_features[:, 2].max()-self.flipped_features[:, 2].min())
        self.flipped_features[:, 3] = (self.flipped_features[:, 3]-self.flipped_features[:, 3].min()) / (self.flipped_features[:, 3].max()-self.flipped_features[:, 3].min())

        print("ΔR min:", dR.min().item())
        print("ΔR max:", dR.max().item())
        
        print("ΔR mean min:", dR_mean.min().item())
        print("ΔR mean max:", dR_mean.max().item())
        
        print("ΔR std min:", dR_std.min().item())
        print("ΔR std max:", dR_std.max().item())
        
        print("Weights (pixel intensity) min:", weights.min().item())
        print("Weights (pixel intensity) max:", weights.max().item())
        
        print("Pixel mean min:", pixel_mean.min().item())
        print("Pixel mean max:", pixel_mean.max().item())
        
        print("Pixel std min:", pixel_std.min().item())
        print("Pixel std max:", pixel_std.max().item())

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        image = self.images[idx]
        flipped_image = self.flipped_images[idx]
        features = self.features[idx]
        flipped_features = self.flipped_features[idx]

        return image, features, flipped_image, flipped_features

