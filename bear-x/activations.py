"""
We are gonna start with simple
and powerful activation function.
Later on I am gonna add more functions
"""
from tensor import Tensor


def relu(inputs: Tensor) -> Tensor:
    for node in range(len(inputs)):
        if inputs[node] > 0:
            continue
        else:
            inputs[node] = 0
    return inputs
