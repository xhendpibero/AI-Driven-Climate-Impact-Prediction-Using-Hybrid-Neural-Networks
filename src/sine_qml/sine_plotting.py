import matplotlib.pyplot as plt

def plot_sine_predictions(X, y_true, y_pred):
    """Plots the true and predicted sine values."""
    plt.figure(figsize=(8, 6))
    plt.plot(X, y_true, label="True Sine Function", color='blue')
    plt.plot(X, y_pred, label="Predicted Sine Function", color='red', linestyle='--')
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()
    plt.show()