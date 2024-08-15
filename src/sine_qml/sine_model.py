import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.sine_qml import sine_circuit

# Initialize the quantum device globally
dev = qml.device("default.qubit", wires=2)

class SineModel(keras.Model):
    def __init__(self, n_layers=3):
        super(SineModel, self).__init__()
        self.params = tf.Variable(np.random.uniform(0, np.pi, (n_layers, 3)), trainable=True)
        self.n_layers = n_layers

    def call(self, inputs):
        """Forward pass through the quantum model."""
        outputs = []
        for x in inputs:
            for i in range(self.n_layers):
                q_results = sine_circuit(self.params[i], x)
                outputs.append(q_results)
        return tf.convert_to_tensor(outputs, dtype=tf.float32)
    
    def train_step(self, data):
        """Custom training step to update quantum circuit parameters."""
        X, y = data
        with tf.GradientTape() as tape:
            predictions = self(X, training=True)
            loss = self.compiled_loss(y, predictions)
        
        gradients = tape.gradient(loss, [self.params])
        self.optimizer.apply_gradients(zip(gradients, [self.params]))
        return {"loss": loss}