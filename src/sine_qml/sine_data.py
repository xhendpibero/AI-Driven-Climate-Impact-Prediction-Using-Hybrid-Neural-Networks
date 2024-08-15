import numpy as np

def generate_sine_data(n_points=100):
    """Generates data for the sine function on the interval [0, 2π]."""
    X = np.linspace(0, 2 * np.pi, n_points)
    y = np.sin(X)
    return X, y