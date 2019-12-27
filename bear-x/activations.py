"""
We are gonna start with simple
and powerful activation function.
Later on I am gonna add more functions
"""
from tensor import Tensor

import numpy as np


def relu(inputs: Tensor) -> Tensor:
    assert type(inputs) == np.ndarray
    inputs = [np.maximum(0, inputs[node]) for node in range(len(inputs))]
    return inputs


def relu_prime(inputs: Tensor) -> Tensor:
    assert type(inputs) == np.ndarray
    inputs = [1 if inputs[node] > 0 else 0 for node in range(len(inputs))]
    return inputs


def sigmoid(inputs: Tensor) -> Tensor:
    assert type(inputs) == np.ndarray
    inputs = [1 / (1 + np.exp(-inputs[node])) for node in range(len(inputs))]
    return inputs


def sigmoid_prime(inputs: Tensor) -> Tensor:
    assert type(inputs) == np.ndarray
    inputs = [sigmoid(node) * (1 - sigmoid(node)) for node in range(len(inputs))]
    return inputs
