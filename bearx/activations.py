from tensor import Tensor
from losses import CrossEntropy

import numpy as np

# TODO: change sigmoid function


def relu(inputs: Tensor) -> Tensor:
    assert type(inputs) == np.ndarray, "Inputs have to be a Tensor(np.array)!"
    return np.maximum(inputs, 0, inputs)


def relu_prime(inputs: Tensor) -> Tensor:
    assert type(inputs) == np.ndarray, "Inputs have to be a Tensor(np.array)!"
    return (inputs > 0).astype(inputs.dtype)


def sigmoid(inputs: Tensor) -> Tensor:
    assert type(inputs) == np.ndarray, "Inputs have to be a Tensor(np.array)!"
    inputs = [1 / (1 + np.exp(-inputs[node])) for node in range(len(inputs))]
    return np.array(inputs)


def sigmoid_prime(inputs: Tensor) -> Tensor:
    assert type(inputs) == np.ndarray, "Inputs have to be a Tensor(np.array)!"
    inputs = [sigmoid(node) * (1 - sigmoid(node))
              for node in range(len(inputs))]
    return np.array(inputs)


def tanh(x: Tensor) -> Tensor:
    # TODO: implement own tanh
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2


def softmax(x: Tensor, axis=-1) -> Tensor:
    if x.ndim == 2:
        y = np.exp(x - np.max(x, axis, keepdims=True))
        return y / np.sum(y, axis, keepdims=True)
    elif x.ndim > 2:
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        s = np.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError("Cannot apply softmax to a tensor that is 1D\n"
                         f"Receiced input: {x}")

"""
def softmax_prime(x: Tensor, y: int):
    probs = softmax(x)
    probs[y] -= 1.0
    return probs
"""

def softmax_prime(x):
    x = softmax(x)
    s = x.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


class Softmax:
    def __init__(self, epsilon=1e-12) -> None:
        self.epsilon = epsilon
        self.ce = CrossEntropy()

    def predict(self, x: Tensor) -> Tensor:
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    def loss(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        Cross entropy
        """
        self.ce.loss(preds, targets, self.epsilon)
        return -np.log(preds[y])
