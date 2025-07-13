# QGAN_4_HEP_PURSUE

This repository contains code for benchmarking **Quantum Generative Adversarial Networks (QGANs)** against classical GANs for **High Energy Physics (HEP)** simulations, specifically jet image generation. This work was conducted under the **PURSUE program with USCMS and Fermilab** under the guidance of Dr. Sergei Gleyzer and Eric Reinhardt.

## 🧠 Motivation

This (Q)GAN seeks to implement novel HEP-motivated metrics (ΔR and energy statistics), tuneable distribution output, feature-based discrimination, and implicit η symmetry from detector geometry. The model incorporates various physical features (signal, η, ϕ, p<sub>T</sub>, mass, \Delta R, and statistical distributions) as codings for the generator, training it to produce a desired distribution of physically informed images. The discriminator is trained on the jet image and the feature codings (label, η, mass, p<sub>T</sub>), allowing it to learn classification of events from low-level detector data.

---

## 🗂️ Dataset

THe primary dataset is the **Jet Mass dataset** found [here](https://data.mendeley.com/datasets/4r4v785rgx/1):

Nachman, Benjamin; de Oliveira, Luke; Paganini, Michela (2017), “Pythia Generated Jet Images for Location Aware Generative Adversarial Network Training”, Mendeley Data, V1, doi: 10.17632/4r4v785rgx.1

- `jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5`
  - Contains: 25×25 calorimeter jet images
  - Labels: signal/background, η, ϕ, p<sub>T</sub>, mass, ΔR, etc.
  - Depends on Pythia, ROOT, FastJet and Python
  - Based on CMS detector simulations

---

## ⚙️ Project Features

- 🌀 **Quantum Layer**: A Pennylane-based `AmplitudeEmbedding` + `RandomLayers` quantum circuit encodes initial features.
- 🔁 **η-Symmetry Enforcement**: Images and features are flipped along η-axis, then the forward pass is averaged with the original to enforce physical invariance.
- 📊 **Statistical Losses**: Custom KL divergence losses over:
  - Mean and std of ΔR (jet radius)
  - Pixel intensity distribution
- 🔢 **Soft Nonzero Pixel Count**: A differentiable regularization loss that encourages characteristic image sparsity.
- 🧪 **Conditional Generation**: Generator conditioned on physics features (e.g. signal, η, p<sub>T</sub>, mass) sampled from KDE of real data.
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
- Input: image & features (label, η, mass, p<sub>T</sub>)
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
- `α`  
- `β`  
- `χ`

---

## 🖼️ Visualizations

- **Generated Jet Samples** at each epoch
- **Loss Curves** for Generator and Discriminator
- **Statistical Matching** for input/output ΔR and pixel stats

---

## 🛠️ How to Run

**Recommended:** HPC CLuster with A100+ GPU

Training will be very slow without a powerful GPU.

```bash
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt

git clone https://github.com/JaySeeDub/QGAN_4_HEP_PURSUE.git <your_directory_name>
cd <your_directory_name>/data
wget https://data.mendeley.com/public-files/datasets/4r4v785rgx/files/132306f6-26f4-4583-8f1b-ccc5ad8da05d/file_downloaded
mv file_downloaded jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5
```
Navigate to Scripts/ (or Notebooks/ for interactive) and run the model.
