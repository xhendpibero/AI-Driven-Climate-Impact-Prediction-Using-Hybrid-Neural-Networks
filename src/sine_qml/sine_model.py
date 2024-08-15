import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from tensorflow import keras

# Initialize the quantum device globally
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

class SineModel(keras.Model):
    def __init__(self, n_layers=3):
        super(SineModel, self).__init__()
        self.params = tf.Variable(np.random.uniform(0, np.pi, (n_layers, 3)), trainable=True)
        self.n_layers = n_layers

    def call(self, inputs):
        """Forward pass through the quantum model."""
        def circuit_apply(x):
            """Applies the quantum circuit for each input."""
            outputs = []
            for i in range(self.n_layers):
                q_result = sine_circuit(self.params[i], x)
                outputs.append(q_result)
            return tf.stack(outputs)
        
        return tf.map_fn(circuit_apply, inputs)
    
    def train_step(self, data):
        """Custom training step to update quantum circuit parameters."""
        X, y = data
        with tf.GradientTape() as tape:
            predictions = self(X, training=True)
            loss = self.compiled_loss(y, predictions)
        
        gradients = tape.gradient(loss, [self.params])
        self.optimizer.apply_gradients(zip(gradients, [self.params]))
        return {"loss": loss}
