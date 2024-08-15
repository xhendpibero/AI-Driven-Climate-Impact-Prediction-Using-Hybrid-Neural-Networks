import numpy as np
import matplotlib.pyplot as plt
from .sine_data import generate_sine_data
from .sine_model import create_qml_model

def main():
    # Generate sine data
    x_values, y_values = generate_sine_data(n_points=100)
    
    # Reshape data to match input shape
    x_values_reshaped = x_values.reshape(-1, 1)
    y_values_reshaped = y_values.reshape(-1, 1)

    # Create the QML model
    qml_model = create_qml_model()

    # Train the model
    qml_model.fit(x_values_reshaped, y_values_reshaped, epochs=100, batch_size=10)

    # Predict and plot the results
    y_pred = qml_model.predict(x_values_reshaped)

    plt.plot(x_values, y_values, label="Original Sine Function")
    plt.plot(x_values, y_pred, label="QML Model Predictions", linestyle='--')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
