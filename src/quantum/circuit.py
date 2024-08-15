import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

# Initialize the quantum device
dev = qml.device("default.qubit", wires=4)
rand_params = np.random.uniform(high=2 * np.pi, size=(1, 4))

@qml.qnode(dev)
def circuit(phi):
    """Quantum circuit for processing image patches."""
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)
    RandomLayers(rand_params, wires=list(range(4)))
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

def quanv(image):
    """Apply the quantum circuit to an image."""
    out = np.zeros((14, 14, 4))
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            q_results = circuit([
                image[j, k, 0],
                image[j, k + 1, 0],
                image[j + 1, k, 0],
                image[j + 1, k + 1, 0]
            ])
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]
    return out