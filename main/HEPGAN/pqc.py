"""
pqc.py — PennyLane PQC feature embedding module
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.templates import RandomLayers, StronglyEntanglingLayers


# ---------------------------------------------------
# Feature embedding circuits
# ---------------------------------------------------
def rl_feature_embedding(f, phi, n_qubits):
    """RandomLayer feature embedding."""
    qml.AngleEmbedding(features=f, wires=range(n_qubits))
    qml.RandomLayers(phi, wires=range(n_qubits), seed=6)
    return qml.state()


def se_feature_embedding(f, phi, n_qubits):
    """StronglyEntanglingLayers feature embedding."""
    qml.AngleEmbedding(features=f, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights=phi, wires=range(n_qubits))
    return qml.state()


# ---------------------------------------------------
# PQC initializer
# ---------------------------------------------------
def InitializePQC(circuit_type: str, n_qubits: int = 8, n_layers: int = 1):
    """
    Create a quantum feature embedding module wrapped in a torch.nn.Module.

    Args:
        circuit_type: \"RandomLayer\" or \"StronglyEntangling\"
        n_qubits: number of qubits in the PQC
        n_layers: number of PQC layers

    Returns:
        QuantumFeatureEmbedding (torch.nn.Module)
    """

    dev = qml.device("default.qubit", wires=n_qubits)

    # ---------------------------------------------
    # Select embedding type + parameter shape
    # ---------------------------------------------
    if circuit_type == "RandomLayer":
        phi_shape = (n_layers, n_qubits * 2)

        def embedding(f, phi):
            return rl_feature_embedding(f, phi, n_qubits)

    elif circuit_type == "StronglyEntangling":
        phi_shape = (n_layers, n_qubits, 3)

        def embedding(f, phi):
            return se_feature_embedding(f, phi, n_qubits)

    else:
        raise ValueError("circuit_type must be 'RandomLayer' or 'StronglyEntangling'")

    # ---------------------------------------------
    # Define the QNode
    # ---------------------------------------------
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnode(f, phi):
        return embedding(f, phi)

    # ------------------------------------------------
    # Torch module wrapper for batching
    # ------------------------------------------------
    class QuantumFeatureEmbedding(nn.Module):
        """
        Wrapper allowing:
            encoded_features = pqc(x_batch)
        Produces shape (B, 256) real-valued feature vectors (for 8 qubits).
        """

        def __init__(self, device="cuda"):
            super().__init__()
            self.device = device

            # Trainable parameters for PQC
            self.phi = nn.Parameter(
                torch.tensor(
                    np.random.uniform(0, 2 * np.pi, phi_shape),
                    dtype=torch.float32
                )
            )

        def forward(self, x_batch: torch.Tensor) -> torch.Tensor:
            """
            Evaluate PQC on a batch of inputs.
            Torch QNodes are not inherently batch-parallel, so loop.

            Args:
                x_batch: (B, n_features)

            Returns:
                (B, circuit_output_dim) — real amplitudes.
            """
            outputs = []
            phi_cpu = self.phi.detach().cpu()

            for x in x_batch:
                out = qnode(x.detach().cpu(), phi_cpu)
                outputs.append(out.real.to(self.device))

            return torch.stack(outputs)

    return QuantumFeatureEmbedding()
