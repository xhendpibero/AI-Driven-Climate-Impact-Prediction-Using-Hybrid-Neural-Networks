import tensorflow as tf
from tensorflow import keras
from .sine_circuit import quantum_circuit

class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(QuantumLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Use tf.map_fn to apply the quantum circuit to each input in the batch
        return tf.map_fn(lambda x: quantum_circuit(x), inputs)

def create_qml_model():
    """Create a hybrid quantum-classical model to learn the sine function."""
    model = keras.models.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(1,)),
        QuantumLayer(),  # Apply the quantum layer
        keras.layers.Dense(1)  # Output layer to predict sine value
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
