import numpy as np

# Function to generate dataset
def generate_sine_data(num_points=100):
    x = np.linspace(0, 2 * np.pi, num_points)
    y = np.sin(x)
    return x, y

# Function to save model parameters
def save_model_params(params, file_path):
    np.save(file_path, params)

# Function to load model parameters
def load_model_params(file_path):
    return np.load(file_path)

# Function to plot predictions
import matplotlib.pyplot as plt

def plot_predictions(x, y_true, y_pred, title="Sine Function Approximation"):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, label="True Function")
    plt.plot(x, y_pred, label="Predicted Function", linestyle='dashed')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.show()