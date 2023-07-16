import numpy as np

class ActivationFunction:
    def __init__(self, function:callable, function_derivative:callable):
        self.function = function
        self.function_derivative = function_derivative

# Leaky Relu
# ----------
def leaky_relu(x, alpha=0.01):
    return np.maximum(x, alpha * x)
def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

# Relu
# ----
def relu(x):
    return np.maximum(x, 0)
def relu_derivative(x):
    return np.ones_like(x)

# Sigmoid
# -------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

# Linear
# ------
def linear(x):
    return x
def linear_derivative(x):
    return 1