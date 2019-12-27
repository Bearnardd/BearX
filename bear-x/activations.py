"""
We are gonna start with simple
and powerful activation function.
Later on I am gonna add more functions
"""
from tensor import Tensor

import numpy as np



def relu(inputs: Tensor) -> Tensor:
    assert type(inputs) == np.ndarray
    outputs = inputs.copy()
    for node in range(len(outputs)):
        if outputs[node] > 0:
            continue
        else:
            outputs[node] = 0
    return outputs 
 
def sigmoid(inputs: Tensor) -> Tensor:
    assert type(inputs) == np.ndarray
    outputs = inputs.copy()
    for node in range(len(outputs)):
        outputs[node] = 1 / (1 + np.exp(-outputs[node]))
    return outputs
