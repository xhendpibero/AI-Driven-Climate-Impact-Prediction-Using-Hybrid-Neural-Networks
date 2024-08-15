import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def sine_circuit(params, x):
    """Quantum circuit to learn the sine function."""
    qml.RX(x, wires=0)
    qml.RY(params[0], wires=0)
    qml.RX(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[2], wires=1)
    return qml.expval(qml.PauliZ(1))