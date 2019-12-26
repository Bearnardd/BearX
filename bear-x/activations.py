"""
We are gonna start with simple
and powerful activation function.
Later on I am gonna add more functions
"""
from tensor import Tensor

import numpy as np


class Activation:
    def calc(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError(
            "Not implementeed in base class!"
        )

class relu(Activation):
    def __getitem__(self):
        return "relu"
 
    @staticmethod
    def calc(inputs: Tensor) -> Tensor:
        assert type(inputs) == np.ndarray
        outputs = inputs.copy()
        for node in range(len(outputs)):
            if outputs[node] > 0:
                continue
            else:
                outputs[node] = 0
        return outputs 

class sigmoid(Activation):
    def __getitem__(self):
        return "sigmoid"

    
    @staticmethod
    def calc(inputs: Tensor) -> Tensor:
        assert type(inputs) == np.ndarray
        outputs = inputs.copy()
        for node in range(len(outputs)):
            outputs[node] = 1 / (1 + np.exp(-outputs[node]))
        return outputs
