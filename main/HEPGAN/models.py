"""
GAN Model Definitions
========================

Clean, linted, modular definitions for:
- Classical Generator / Discriminator
- PQC Generator / Discriminator
- Unified model initialization
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import Tuple

from HEPGAN.pqc import InitializePQC
from HEPGAN.helper import GaussianNoise, soft_threshold


# -------------------------------------------------------------------------
#  Model Initialization
# -------------------------------------------------------------------------

def InitializeModel(model_type: str, sigma: float):
    """
    Return (generator, discriminator) depending on model_type.
    """

    if model_type == "classical":
        gen = Generator(sigma=sigma)
        disc = Discriminator()
        return gen, disc

    elif model_type == "diffusion":
        gen = Diffusion(sigma=sigma)
        disc = Discriminator()
        return gen, disc

    # PQC variants
    elif model_type in ("RandomLayer", "StronglyEntangling"):
        pqc_layer = InitializePQC(model_type)
        gen = PQCGenerator(pqc_layer, sigma=sigma)
        disc = PQCDiscriminator(pqc_layer)
        return gen, disc

    raise ValueError(f"Invalid model choice: {model_type}")


# -------------------------------------------------------------------------
#  Classical Generator
# -------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, sigma: float = 0.1):
        super().__init__()

        self.noise = GaussianNoise(sigma=sigma)

        # Conditioning features → feature vector
        self.feature_gen = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            self.noise
        )

        # 256 × 1 × 1 → 1 × 16 × 16 image
        self.image_gen = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout(0.2),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z_feat: torch.Tensor) -> torch.Tensor:
        features = self.feature_gen(z_feat)
        img = features.view(-1, 256, 1, 1)
        img = self.image_gen(img)
        img = soft_threshold(img, threshold=0.001, sharpness=1000.0)
        return img


# -------------------------------------------------------------------------
#  Classical Discriminator
# -------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Image encoder
        self.image_encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Flatten()
        )

        # Cond features: 4 → 128
        self.feature_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )

        # Inspect dimensions dynamically
        with torch.no_grad():
            dummy_img = torch.zeros(1, 1, 16, 16)
            img_dim = self.image_encoder(dummy_img).shape[1]

        flat_dim = img_dim + 128

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        img_enc = self.image_encoder(img)
        feat_enc = self.feature_encoder(features)
        x = torch.cat((img_enc, feat_enc), dim=1)
        return self.classifier(x)


# -------------------------------------------------------------------------
#  PQC Generator
# -------------------------------------------------------------------------

class PQCGenerator(nn.Module):
    def __init__(self, pqc_encoder, sigma: float):
        super().__init__()

        self.feature_encoder = pqc_encoder
        self.noise = GaussianNoise(sigma=sigma)

        self.classifier = nn.Sequential(
            self.noise,
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.feature_encoder(features).float()
        return self.classifier(encoded)


# -------------------------------------------------------------------------
#  PQC Discriminator
# -------------------------------------------------------------------------

class PQCDiscriminator(nn.Module):
    def __init__(self, pqc_encoder):
        super().__init__()

        self.feature_encoder = pqc_encoder

        self.image_encoder = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Flatten()
        )

        with torch.no_grad():
            dummy_img = torch.zeros(1, 1, 16, 16)
            img_dim = self.image_encoder(dummy_img).shape[1]

            dummy_feat = torch.zeros(1, 4)
            feat_dim = self.feature_encoder(dummy_feat).shape[1]

        flat_dim = img_dim + feat_dim

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        img_enc = self.image_encoder(img)
        feat_enc = self.feature_encoder(features)
        x = torch.cat((img_enc, feat_enc), dim=1)
        return self.classifier(x)
