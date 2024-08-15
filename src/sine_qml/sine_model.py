import pennylane as qml
from pennylane import numpy as pnp

# Define the quantum circuit
def create_quantum_circuit(params):
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(params):
        qml.RX(params[0], wires=0)
        qml.RZ(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    return circuit

# Define the cost function
def cost_function(params, x, y):
    # Ensure params is a Pennylane-compatible array
    params = pnp.asarray(params, dtype=float)
    
    # Ensure x is a Pennylane-compatible array
    x = pnp.asarray(x, dtype=float)
    y = pnp.asarray(y, dtype=float)

    circuit = create_quantum_circuit(params)
    predictions = pnp.array([circuit([x_i, params[1]]) for x_i in x])
    
    return pnp.mean((predictions - y) ** 2)

# Define the training function
def train_quantum_model(x, y, num_epochs=1000, lr=0.1):
    params = pnp.array([0.1, 0.1], dtype=float)  # Initial parameters
    
    opt = qml.GradientDescentOptimizer(lr)
    
    for epoch in range(num_epochs):
        # Use a lambda function to pass params to the cost_function
        params = opt.step(lambda p: cost_function(p, x, y), params)
        if epoch % 100 == 0:
            # Ensure the cost function output is a scalar
            cost = cost_function(params, x, y)
            print(f"Epoch {epoch}: Cost = {cost}")
    
    return params  # Make sure to return the final optimized parameters


def predict_quantum_model(params, x):
    if params is None:
        raise ValueError("The parameter array 'params' is None. Ensure that the training function is returning the parameters correctly.")
    
    circuit = create_quantum_circuit(params)
    predictions = pnp.array([circuit([x_i, params[1]]) for x_i in x])
    return predictions

