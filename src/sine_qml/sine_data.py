import numpy as np

def generate_sine_data(n_points=100):
    """Generates data points for the sine function on [0, 2π]."""
    x_values = np.linspace(0, 2 * np.pi, n_points)
    y_values = np.sin(x_values)
    return x_values, y_values
