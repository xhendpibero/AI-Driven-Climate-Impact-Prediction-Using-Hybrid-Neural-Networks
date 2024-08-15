import os
import numpy as np

from src.sine_qml.sine_model import predict_quantum_model, train_quantum_model

# Define the absolute path to the data directory
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../saved_data'))

# Load data
x = np.load(os.path.join(data_dir, 'x.npy'))  # Load the actual data from x.npy
y = np.load(os.path.join(data_dir, 'y.npy'))  # Load the actual data from y.npy

# Train the model
params = train_quantum_model(x, y, num_epochs=1000, lr=0.1)

# Save trained parameters
np.save(os.path.join(data_dir, 'params.npy'), params)

# Predict using the trained model
predictions = predict_quantum_model(params, x)

# Save predictions
np.save(os.path.join(data_dir, 'predictions.npy'), predictions)
