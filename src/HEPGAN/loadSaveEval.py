"""
loadSaveEval.py

Clean utilities for:
- saving models
- loading models
- evaluating generated samples

"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Union

import torch
from torch.utils.data import DataLoader

# External diagnostics functions
from HEPGAN.diagnostics import test_generated_samples, test_smote_samples


# ---------------------------------------------------------
#  MODEL SAVE / LOAD
# ---------------------------------------------------------

def save(model: torch.nn.Module, save_path: str | None = None) -> str:
    """
    Save a PyTorch model state_dict with timestamped filename.

    Args:
        model: torch.nn.Module
        save_path: Optional custom path

    Returns:
        Full path to saved file
    """
    os.makedirs("models", exist_ok=True)

    generator, discriminator = model
    
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_losses": g_losses,
            "d_losses": d_losses,
            "validity_hist": validity_hist,
            "stat_hist": stat_hist,
            "nnz_hist": nnz_hist,
            "loss_dR_mean_hist": loss_dR_mean_hist,
            "loss_dR_std_hist": loss_dR_std_hist,
            "loss_pix_mean_hist": loss_pix_mean_hist,
            "loss_pix_std_hist": loss_pix_std_hist,
            "stats_dict": stats_dict,
        },
        save_path,
    )

    print(f"Saved checkpoint to {save_path}")
    


def load(model: (torch.nn.Module, torch.nn.Module), load_path: str, device: str) -> torch.nn.Module:
    """
    Load a model's weights.

    Args:
        model: (generator, discriminator) torch.nn.Modules to load into
        load_path: path to .pt file

    Returns:
        The model with loaded weights
    """
        
    checkpoint = torch.load(load_path, map_location = device)

    generator, discriminator = model
    
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    print(f"Loaded model from {load_path}")


# ---------------------------------------------------------
#  MODEL EVALUATION
# ---------------------------------------------------------

def eval_model(
    model: (torch.nn.Module, torch.nn.Module),
    dataset,
    sampleloader: Union[Dict, DataLoader],
    batch_size: int = 100_000,
):
    """
    Evaluate generated samples, dispatching automatically based on sampler type.

    Args:
        generator: The generator model
        discriminator: The discriminator model
        dataset: Real dataset object (input to diagnostics)
        sampleloader: Either a dict of KDE samplers OR a DataLoader (SMOTE)
        batch_size: Optional batch size for evaluation

    Returns:
        Whatever the diagnostics function returns
    """
    
    generator = model[0]
    discriminator = model[1]

    # KDE-based sampler (dict)
    if isinstance(sampleloader, dict):
        print("Running KDE-based diagnostics...")
        return test_generated_samples(
            generator,
            discriminator,
            dataset,
            sampleloader,
            batch_size=batch_size,
        )

    # SMOTE DataLoader
    if isinstance(sampleloader, DataLoader):
        print("Running SMOTE-based diagnostics...")
        return test_smote_samples(
            generator,
            discriminator,
            dataset,
            sampleloader,
            batch_size=batch_size,
        )

    raise TypeError(
        f"eval_model() expected dict or DataLoader, got {type(sampleloader)}"
    )
