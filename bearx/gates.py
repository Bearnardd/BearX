from bearx.tensor import Tensor

import numpy as np


class AddGate:
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        forward pass in addition gate
        """
        return x1 + x2

    def backward(self, x1: Tensor, x2: Tensor, dz: Tensor):
        """
        backward pass in addition gate
        """
        dx1 = dz * np.ones_like(x1)
        dx2 = dz * np.ones_like(x2)
        return dx1, dx2
        

class MultiplyGate:
    def forward(self, W: Tensor, x: Tensor) -> Tensor:
        return np.dot(W, x)

    def backward(self, W: Tensor, x: Tensor, dz: Tensor) -> Tensor:
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))
        dx = np.dot(np.transpose(W), dz)
        return dW, dx
