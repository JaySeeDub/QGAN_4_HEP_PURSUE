#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Executable training script for HEPGAN/QGAN models.

All configuration is given as command-line arguments:

    python train_hepgan.py \
        --jet-images ../data/jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5 \
        --datasets-path datasets.pt \
        --train-on signal \
        --sampler kde \
        --model-type classical \
        --sigma 0.1 \
        --batch-size 128 \
        --epochs 10 \
        --lr 0.001 \
        --debug 1
"""

import argparse
import torch
from torch.utils.data import DataLoader
from h5py import File as HDF5File

# --------------------------------------------------------
# Local imports (your modules)
# --------------------------------------------------------
from HEPGAN.train import train
from HEPGAN.models import InitializeModel
from HEPGAN.loadSaveEval import eval_model, save, load


# ========================================================
#                     ARGUMENT PARSER
# ========================================================

def build_parser():
    parser = argparse.ArgumentParser(description="Train HEPGAN/QGAN models")

    parser.add_argument("--jet-images", type=str, required=True,
                        help="Path to Jet Images HDF5 file")

    parser.add_argument("--datasets-path", type=str, default="datasets.pt",
                        help="Path to datasets.pt containing pre-built datasets")

    parser.add_argument("--train-on", type=str, default="signal",
                        choices=["signal", "background", "both"],
                        help="Dataset split to use")

    parser.add_argument("--sampler", type=str, default="kde",
                        choices=["kde", "smote"],
                        help="Sampling method for conditioning features")

    parser.add_argument("--model-type", type=str, default="classical",
                        choices=["classical", "StronglyEntangling", "RandomLayer"],
                        help="Type of model to initialize")

    parser.add_argument("--sigma", type=float, default=0.1,
                        help="Gaussian noise level for quantum layers (if applicable)")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--debug", type=int, default=0,
                        help="Enable debug printouts")

    parser.add_argument("--save-model", action="store_true",
                        help="Save final model after training")

    parser.add_argument("--load-model", type=str, default="",
                        help="Optional: load checkpoint before training")

    return parser


# ========================================================
#                           MAIN
# ========================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ---------------------------------------------------
    # Device
    # ---------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---------------------------------------------------
    # Load Jet Images HDF5
    # ---------------------------------------------------
    print(f"[INFO] Loading HDF5 jet dataset from {args.jet_images}")
    jet_mass_data = HDF5File(args.jet_images, "r")
    print("Keys:", list(jet_mass_data.keys()))
    print("Image shape:", jet_mass_data["image"].shape)

    # ---------------------------------------------------
    # Load pre-constructed dataset / samplers
    # ---------------------------------------------------
    print(f"[INFO] Loading datasets from {args.datasets_path}")
    saved = torch.load(args.datasets_path, map_location="cpu")["datasets"]

    dataset = saved[f"dataset_{args.train_on}"]
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True)

    if args.sampler == "smote":
        sample_dataset = saved[f"smoteSampler_{args.train_on}"]
        sampleloader = DataLoader(sample_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True)
    else:
        sampleloader = saved[f"kdeSampler_{args.train_on}"]

    print(f"[INFO] Number of samples: {len(dataset)}")
    print("Image shape:", dataset.images.shape)
    print("Feature shape:", dataset.features.shape)

    # ---------------------------------------------------
    # Initialize Model
    # ---------------------------------------------------
    print(f"[INFO] Initializing model: {args.model_type}")
    model = InitializeModel(model_type=args.model_type, sigma=args.sigma)

    # Optional: load existing checkpoint
    if args.load_model:
        print(f"[INFO] Loading checkpoint: {args.load_model}")
        model = load(model=model, load_path=args.load_model)

    # ---------------------------------------------------
    # Train
    # ---------------------------------------------------
    print("[INFO] Starting training…")
    train(
        model=model,
        dataloader=dataloader,
        sampleloader=sampleloader,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        debug=bool(args.debug),
    )

    # ---------------------------------------------------
    # Evaluate
    # ---------------------------------------------------
    print("[INFO] Evaluating model…")
    eval_model(model=model, dataset=dataset, sampleloader=sampleloader)

    # ---------------------------------------------------
    # Save final model (optional)
    # ---------------------------------------------------
    if args.save_model:
        print("[INFO] Saving final model…")
        save(model=model, save_path=None)

    print("[INFO] Done.")


# ========================================================
# ENTRY POINT
# ========================================================
if __name__ == "__main__":
    main()
