import numpy as np
import tensorflow as tf
from src.sine_qml.sine_data import generate_sine_data
from src.sine_qml.sine_model import SineModel
from src.sine_qml.sine_plotting import plot_sine_predictions

def main():
    # Generate data
    X, y = generate_sine_data(n_points=100)
    
    # Convert data to TensorFlow tensors
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    
    # Initialize and compile the model
    model = SineModel()
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X, y, epochs=100, batch_size=10, verbose=1)
    
    # Predict using the trained model
    y_pred = model(X)
    
    # Plot the results
    plot_sine_predictions(X.numpy(), y.numpy(), y_pred.numpy())

if __name__ == "__main__":
    main()