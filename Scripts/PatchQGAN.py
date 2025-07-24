#!/usr/bin/env python
# coding: utf-8

# In[1]:


## This is adapted from the ML4SCI Group's QGAN project by Luis Guadarama for the jet-mass dataset. The patch method is a possible (albeit unfavorable) option for replacing the classical convolutional layers
# https://github.com/ML4SCI/QMLHEP/tree/main/Quantum_GAN_for_HEP_Luis_Rey_Guadarrama/demos/partial_measure_qGAN

# Library imports
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pennylane as qml
import h5py
from IPython.display import Markdown, display, clear_output
from datetime import date
import json
import numpy as np
from h5py import File as HDF5File
from scipy.stats import gaussian_kde
import os


# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from itertools import product

import numpy as np
from scipy.stats import wasserstein_distance, entropy
from scipy.special import rel_entr
from scipy.linalg import sqrtm



# Set the random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# In[2]:


def FID(real_data, generated_data):
    mu_r = np.mean(real_data, axis=0)
    mu_g = np.mean(generated_data, axis=0)
    C_r = np.cov(real_data, rowvar=False)
    C_g = np.cov(generated_data, rowvar=False)

    mean_diff = mu_r - mu_g
    cov_mean = sqrtm(C_r.dot(C_g))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    distance = mean_diff.dot(mean_diff) + np.trace(C_r + C_g - 2*cov_mean)
    return distance


# In[3]:


test_id = '06'
num_qubits = 7
num_aux_qubits = 2
circuit_depth = 10
num_generators = 2
rotations = ['Y']
generator_lr = 0.001
discriminator_lr = 0.1
batch_size = 1
num_samples = 512
num_epochs = 30
y = 0.3
channel = 'ECAL'
optimizer= 'SGD'
resolution= '8x8'


# In[4]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
jet_images_path = '../data/jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5'
jet_mass_data =h5py.File(jet_images_path, 'r')
images = jet_mass_data['image'][:num_samples, 9:17, 9:17]

print(jet_mass_data.keys())
print(jet_mass_data['image'].shape)


# In[5]:


gluon_ECAL_overlay = np.mean(images, axis=0)
total_energy = np.sum(images, axis=(1,2))
particles_energy_deposits = images[images > 0]


# In[6]:


fig, axs = plt.subplots(1, 2, figsize=(11, 5))

im1 = axs[0].imshow(images[0], cmap="Blues", norm=mcolors.LogNorm())
axs[0].set_title("ECAL jet")

axs[1].imshow(gluon_ECAL_overlay, cmap="Blues", norm=mcolors.LogNorm())
axs[1].set_title("ECAL overlay")

for ax in axs.flat:
    ax.set(xlabel="i$\phi '$", ylabel="i$\eta '$")

cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

plt.show()


# In[7]:


fig, axs = plt.subplots(1, 2, figsize=(15, 5))

axs[0].hist(total_energy, bins=30, color="royalblue")
axs[0].set_title("ECAL Total Energy Deposits per jet", fontsize=15)
axs[0].set_xlabel("total energy")
axs[0].set_ylabel("jet count")

axs[1].hist(particles_energy_deposits, range=(0.1, 2), bins=15, color="royalblue")
axs[1].set_title("ECAL Particle Energy Deposits", fontsize=15)
axs[1].set_xlabel("particle energy")
axs[1].set_ylabel("jet particle")

plt.show()


# In[8]:


class QuarkDataset(Dataset):
    def __init__(self, image):
        self.data = torch.tensor(image, dtype=torch.float32) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# In[9]:


# Create the data loader instance
dataset = QuarkDataset(images.reshape(num_samples, 64))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# In[10]:


class Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(64, 128),
            nn.ReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(128, 32),
            nn.ReLU(),
            # Second hidden layer (16 -> output)
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# In[11]:


dev = qml.device("default.qubit", wires=num_qubits)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@qml.qnode(dev, diff_method="backprop")
def quantum_circuit(noise, weights, rotations):
    weights = weights.reshape(circuit_depth, num_qubits, len(rotations))

    # Initialise latent vectors
    for i in range(num_qubits):
        qml.RY(noise[i], wires=i)

    # Repeated layer
    for i in range(circuit_depth):
         # Parameterised layer
        for q in range(num_qubits):
            for idx, rotation in enumerate(rotations):
                if rotation == "X":
                    qml.RX(weights[i][q][idx], wires=q)
                elif rotation == "Y":
                    qml.RY(weights[i][q][idx], wires=q)
                elif rotation == "Z":
                    qml.RZ(weights[i][q][idx], wires=q)

        # Control Z gates
        for y in range(num_qubits - 1):
            qml.CZ(wires=[y, y + 1])
        qml.CZ(wires=[num_qubits - 1, 0])

    return qml.probs(wires=list(range(num_qubits)))


# In[12]:


weights = torch.rand(num_qubits*circuit_depth*len(rotations))
noise = torch.rand(8, num_qubits, device=device) * np.pi / 2


# In[13]:


qml.draw_mpl(quantum_circuit)(noise, weights, rotations)
plt.show()


# In[14]:


def partial_measure(noise, weights, rotations):
    # Non-linear Transform
    probs = quantum_circuit(noise, weights, rotations)
    probsgiven0 = probs[: (2 ** (num_qubits - num_aux_qubits))]

    # Post-Processing
    probsgiven = probsgiven0 / y
    probsgiven[probsgiven < 0.001] = 0
    return probsgiven


# In[15]:


class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, q_delta=1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()


        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(num_qubits*circuit_depth*len(rotations)), requires_grad=True)
                for _ in range(n_generators)
            ]
        )

        self.n_generators = n_generators

    def forward(self, x, rotations):
        # Size of each sub-generator output
        patch_size = 2 ** (num_qubits - num_aux_qubits)

        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        images = torch.Tensor(x.size(0), 0).to(device)

        # Iterate over all sub-generators
        for params in self.q_params:

            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size).to(device)
            for elem in x:
                q_out = partial_measure(elem, params, rotations).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            # Each batch of patches is concatenated with each other to create a batch of images
            images = torch.cat((images, patches), 1)
            


        return images 


# In[16]:


def plot_training_progress():
    # we don't plot if we don't have enough data
    if len(rms_error) < 2:
        return

    clear_output(wait=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 6))

    # Metric 1
    ax1.set_title("RMSE", fontsize=15)
    ax1.plot(epochs, rms_error, color="royalblue", linewidth=3)
    ax1.set_xlabel("Epoch")
    ax1.set_yscale("log")
    ax1.grid()

    # Metric 2
    ax2.set_title("FID", fontsize=15)
    ax2.plot(epochs, fid, color="cornflowerblue", linewidth=3)
    ax2.set_xlabel("Epoch")
    ax2.set_yscale("log")
    ax2.grid()

    # Generated distribution
    im = ax3.imshow(gen_ECAL_overlay, cmap='Blues', aspect='auto', norm=mcolors.LogNorm())
    ax3.set_title('Generated ECAL overlay', fontsize=15)

    fig.colorbar(im, ax=ax3)

    plt.suptitle(f"Epoch {counter}", fontsize=25)
    plt.show()


# In[17]:


discriminator = Discriminator().to(device)
generator = PatchQuantumGenerator(num_generators).to(device)

# Binary cross entropy
criterion = nn.BCELoss()

# Optimizers
optD = optim.SGD(discriminator.parameters(), lr=discriminator_lr)
optG = optim.SGD(generator.parameters(), lr=generator_lr)

real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

# Iteration counter
counter = 0

# plot lists
rms_error = []
fid = []
epochs = []
disc_loss = []
gen_loss = []


# In[24]:


while counter < num_epochs:

    noise = torch.rand(num_samples, num_qubits, device=device)* np.pi / 2
    gen_ECAL_overlay = np.mean(generator(noise, rotations).detach().cpu().numpy().reshape(num_samples, 8, 8), axis=0)
    fd = FID(gluon_ECAL_overlay, gen_ECAL_overlay)
    rms = np.sqrt(np.mean((gluon_ECAL_overlay - gen_ECAL_overlay) ** 2))
    fid.append(fd)
    rms_error.append(rms)
    epochs.append(counter)

    plot_training_progress()
    
    for i, data in enumerate(dataloader):


        # Data for training the discriminator
        #data = data.reshape(-1, image_size * image_size)
        real_data = data.to(device)

        # Noise follwing a uniform distribution in range [0,pi/2)
        noise = torch.rand(batch_size, num_qubits, device=device) * np.pi / 2
        fake_data = generator(noise, rotations)

        # Training the discriminator
        discriminator.zero_grad()
        outD_real = discriminator(real_data).view(-1)
        outD_fake = discriminator(fake_data.detach().cuda()).view(-1)

        errD_real = criterion(outD_real, real_labels)
        errD_fake = criterion(outD_fake, fake_labels)
        # Propagate gradients
        errD_real.backward()
        errD_fake.backward()

        errD = errD_real + errD_fake
        optD.step()

        # Training the generator
        generator.zero_grad()
        outD_fake = discriminator(fake_data).view(-1)
        errG = criterion(outD_fake, real_labels)
        errG.backward()
        optG.step()
            
    counter += 1
    disc_loss.append(errD)
    gen_loss.append(errG)
    print(f"Epoch {counter}: [Disc_Error = {errD}] [Gen_Error = {errG}]")


# In[25]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 6))

# Metric 1
ax1.set_title("Relative Entropy", fontsize=15)
ax1.plot(epochs, rms_error, color="royalblue", linewidth=3)
ax1.set_xlabel("Epoch")
#ax1.set_yscale("log")
ax1.grid()

# Metric 2
ax2.set_title("FID", fontsize=15)
ax2.plot(epochs, fid, color="cornflowerblue", linewidth=3)
ax2.set_xlabel("Epoch")
#ax2.set_yscale("log")
ax2.grid()

# Generated distribution
im = ax3.imshow(gen_ECAL_overlay, cmap='Blues', aspect='auto', norm=mcolors.LogNorm())
ax3.set_title('Generated ECAL overlay', fontsize=15)

fig.colorbar(im, ax=ax3)

plt.suptitle(f"Epoch {counter}", fontsize=25)
plt.show()


# In[26]:


generated_jets = generator(torch.rand(num_samples, num_qubits, device=device) * np.pi / 2, rotations).detach().numpy().reshape(num_samples, 8, 8)
fig, axs = plt.subplots(2, 3, figsize=(13, 7.5)) 

for i, j in product(range(2), range(3)):
    jet = np.random.randint(0, 300)
    im1 = axs[i, j].imshow(generated_jets[jet], cmap="Blues", norm=mcolors.LogNorm())
    axs[i, j].set_xlabel("Energy Deposits")
    axs[i, j].set_ylabel("jet count")

for ax in axs.flat:
    ax.set(xlabel="i$\phi '$", ylabel="i$\eta '$")

cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

plt.suptitle(f"Generated Jets (ECAL)", fontsize=20)
plt.show()


# In[ ]:


fig, axs = plt.subplots(2, 3, figsize=(13, 7.5)) 

for i, j in product(range(2), range(3)):
    jet = np.random.randint(0, 300)
    im1 = axs[i, j].imshow(all_gluon_8x8[jet], cmap="Blues", norm=mcolors.LogNorm())
    axs[i, j].set_xlabel("Energy Deposits")
    axs[i, j].set_ylabel("jet count")

for ax in axs.flat:
    ax.set(xlabel="i$\phi '$", ylabel="i$\eta '$")

cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

plt.suptitle(f"Real Jets (ECAL)", fontsize=20)
plt.show()


# In[ ]:


fig, axs =  plt.subplots(1, 2, figsize=(15, 7)) 
n = np.random.randint(0, 500)

im1 = axs[0].imshow(all_gluon_8x8[n], cmap="Blues", norm=mcolors.LogNorm())
axs[0].set_title("real ECAL jet", fontsize=15)

im2 = axs[1].imshow(generated_jets[n], cmap='Blues', norm=mcolors.LogNorm())
axs[1].set_title("generated ECAL jet", fontsize=15)

for ax in axs.flat:
    ax.set(xlabel="i$\phi '$", ylabel="i$\eta '$")

cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)


# In[ ]:


gen_jets_image = np.mean(generated_jets, axis=0)
real_jets_image = np.mean(all_gluon_8x8, axis=0)

fig, axs =  plt.subplots(1, 2, figsize=(15, 7)) 

im1 = axs[0].imshow(real_jets_image, cmap="Blues", norm=mcolors.LogNorm())
axs[0].set_title("real ECAL overlay", fontsize=15)

im2 = axs[1].imshow(gen_jets_image, cmap='Blues', norm=mcolors.LogNorm())
axs[1].set_title("generated ECAL overlay", fontsize=15)

for ax in axs.flat:
    ax.set(xlabel="i$\phi '$", ylabel="i$\eta '$")

cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)


# In[ ]:


plt.hist(gen_total_energy, range=(0, 2), bins=30, color="royalblue", label="generated", histtype="step", linewidth=2)
plt.hist(gen_total_energy, range=(0, 2), bins=30, color="royalblue", alpha=0.5)
plt.hist(total_energy, range=(0, 2), bins=30, color="plum", label="real", histtype="step", linewidth=2)
plt.hist(total_energy, range=(0, 2), bins=30, color="plum", alpha=0.5)
plt.title("Total ECAL Energy deposits per jet", fontsize=15)
plt.xlabel("total energy")
plt.ylabel("jet count")
plt.legend()
plt.show()


# In[ ]:


gen_particles_energy_deposits = generated_jets[generated_jets > 0]

plt.hist(gen_particles_energy_deposits, range=(0.002, 0.05), bins=30, color="royalblue", label="generated", histtype="step", linewidth=2)
plt.hist(gen_particles_energy_deposits, range=(0.002, 0.05), bins=30, color="royalblue", alpha=0.5)
plt.hist(particles_energy_deposits, range=(0, 0.05), bins=30, color="plum", label="real", histtype="step", linewidth=2)
plt.hist(particles_energy_deposits, range=(0, 0.05), bins=30, color="plum", alpha=0.5)
plt.title("Particle ECAL Energy deposits", fontsize=15)
plt.xlabel("total energy")
plt.ylabel("particle count")
plt.legend()
plt.show()


# In[ ]:




