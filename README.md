# QGAN_4_HEP_PURSUE
Development repo for benchmarking classical vs quantum generative adversarial networks for USCMS x Fermilab. The goal of this project is described 

## Overview
Building off of the physics-oriented motivation, this GAN implements novel physically-based loss functions (\Delta R & p_T statistics), arbitrary distribution output, feature-based discrimination, and detector \eta symmetry. The main dataset explored is from the jet mass dataset found here (https://data.mendeley.com/datasets/4r4v785rgx/1). The model incorporates various physical features (signal, \eta, \phi, \p_T, \Delta R, and statistical distributions) as codings for the generator, allowing output of arbitrary distributions of physically informed images. The discriminator is trained on the jet image and the first 4 codings, allowing classification of events from low-level detector data.

## Classical GAN

## QGAN

## \eta Symmetry

## Statistical Codings

## 


# QGAN_4_HEP_PURSUE

This repository contains code and experiments for benchmarking **Quantum Generative Adversarial Networks (QGANs)** against classical GANs for **High Energy Physics (HEP)** simulations, specifically jet image generation. This work is part of a research effort under the **PURSUE program with USCMS and Fermilab**.

## 🧠 Motivation

HEP data is rich in physics-informed structure—jets from particle collisions exhibit specific symmetries and statistical behaviors. Traditional GANs struggle to reproduce these complex, structured distributions. Our QGAN integrates **quantum circuits**, **physically-informed loss functions**, and **η-symmetry constraints** to produce more realistic jet simulations.

---

## 🗂️ Dataset

We use the **Jet Mass dataset** from [Mendeley Data](https://data.mendeley.com/datasets/4r4v785rgx/1):

- `jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5`
  - Contains: 25×25 calorimeter jet images
  - Labels: signal/background + η, ϕ, p<sub>T</sub>, mass
  - Used in CMS detector simulations

---

## ⚙️ Features

- 🌀 **Quantum Layer**: A Pennylane-based `AmplitudeEmbedding` + `RandomLayers` quantum circuit processes conditioning features.
- 🔁 **η-Symmetry Enforcement**: Images and features are flipped along η-axis to enforce physical invariance.
- 📊 **Statistical Losses**: Custom KL divergence losses over:
  - Mean and std of ΔR (jet radius)
  - Pixel intensity distribution
- 🔢 **Soft Nonzero Pixel Count**: A differentiable regularization loss that encourages sparsity.
- 🧪 **Conditional Generation**: Generator conditioned on physics features (e.g. η, p<sub>T</sub>, mass, etc.) sampled using KDE.
- ✅ **Discriminator**: Jointly encodes images and physics features to classify real vs generated data.

---

## 🏗️ Architecture

### Generator
- Input: 9D feature vector (label + physics)
- Processing:
  - Quantum circuit → 256D state vector
  - Transposed CNN layers upsample to 16×16
- Output: 16×16 jet image

### Discriminator
- Input: (image, first 4 features)
- Encodes:
  - Image via CNN layers
  - Features via MLP
- Output: Real/fake classification (sigmoid)

---

## 🔬 Loss Functions

| Term | Description |
|------|-------------|
| `BCELoss` | Real/fake classification |
| `StatLoss` | KL divergence over ΔR / pixel stats |
| `NNZLoss` | MSE over soft nonzero pixel count |
| `TotalLoss` | `α * BCE + β * NNZ + χ * Stat` |

Tuneable Hyperparameters:  
- `α = 0.05`  
- `β = 1e-5`  
- `χ = 1e-4`

---

## 🖼️ Visualizations

- **Generated Jet Samples** at each epoch
- **Loss Curves** for Generator and Discriminator
- **Statistical Matching** for input/output ΔR and pixel stats

---

## 🛠️ How to Run

### Requirements
- Python 3.10+
- PyTorch
- Pennylane
- h5py, scikit-learn, matplotlib, etc.

```bash
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
