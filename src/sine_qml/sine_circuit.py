import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def quantum_circuit(x):
    """Quantum circuit to encode the input and apply a quantum layer."""
    qml.RY(x, wires=0)  # Encode input x using RY rotation
    qml.RX(0.5, wires=0)  # Apply a quantum operation
    return qml.expval(qml.PauliZ(0))  # Expectation value as output