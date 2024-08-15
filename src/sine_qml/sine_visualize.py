import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../saved_data'))

# Load data
x = np.load(os.path.join(data_dir, 'x.npy'))
y = np.load(os.path.join(data_dir, 'y.npy'))
predictions = np.load(os.path.join(data_dir, 'predictions.npy'))

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='True Sine Function')
plt.plot(x, predictions, label='Predicted by QML Model', linestyle='dashed')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Function vs. Quantum Model Predictions')
plt.show()