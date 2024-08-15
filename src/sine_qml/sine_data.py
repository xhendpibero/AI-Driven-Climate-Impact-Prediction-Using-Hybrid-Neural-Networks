import os
import numpy as np

def prepare_sine_data(num_points=100):
    # Discretize the interval [0, 2π]
    x = np.linspace(0, 2 * np.pi, num_points)
    # Compute the sine values
    y = np.sin(x)
    return x, y

if __name__ == "__main__":
    x, y = prepare_sine_data()
    
    # Define the target directory
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../saved_data'))
    
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Save the data to the correct directory
    np.save(os.path.join(target_dir, 'x.npy'), x)
    np.save(os.path.join(target_dir, 'y.npy'), y)
    print("Data files created: x.npy, y.npy in saved_data/")