from typing import Dict
from tensor import Tensor
import numpy as np
import timeit

from activations import *


class Layer:
    def __init__(self, **kwargs):
        """
        Layer base class
        """
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def __getitem__(self):
        raise NotImplementedError(
            "Function not implemented in base class!"
        )

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError(
            "Function not implemented in base class!"
        )

    def back_prop(self, grad: Tensor) -> Tensor:
        raise NotImplementedError(
            "Function not implemented in base class!"
        )


class Linear(Layer):
    """
    Basic Linear layer.
    Convets inputs as shown below:
    output = inputs * weights + bias
    output = activation_function(output)
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation=None,
                 **kwargs):
        super(Linear, self).__init__(**kwargs)
        # TODO: add activations
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        # TODO: add kwargs
        allowed_kwargs = {
            "weight_initializer"
        }
        for kwarg in kwargs:
            if kwargs not in allowed_kwargs:
                raise TypeError(f"Keyword argument not understood: {kwarg}")
        self.weight_initializer = kwargs.get('weight_initializer', None)

        if self.weight_initializer is None:
            self.params["W"] = np.random.rand(
                self.in_features, self.out_features) * 0.1
            self.params["b"] = np.random.rand(
                self.out_features,) * 0.1

    def __getitem__(self):
        """
        Return information about layer
        """
        item = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "activation": self.activation.__name__ if self.activation is not None else "None"
        }
        return item

    def forward(self, inputs: Tensor) -> Tensor:
        """
        element wise multiplication and addition
        :param: inputs: Tensor - input data
        :return: output matrix with activation function applied
        """
        # both methods give the same results
        # output = np.add(np.multiply(inputs, self.params["W"]), self.params["b"])[0]
        self.inputs = inputs
        output = inputs @ self.params["W"] + self.params["b"]
        if self.activation:
            return self.activation(output)
        return output

    def back_propagation(self, gradient: Tensor) -> Tensor:
        self.grads["b"] = np.sum(gradient, axis=0)
        self.grads["w"] = self.inputs.T @ gradient
        return gradient @ self.params["w"].T


class Activation(Layer):
    def __init__(self, activation, activation_prime) -> None:
        super(Activation, self).__init__()
        self.activation= activation 
        self.activation_prime = activation_prime

    def __getitem__(self):
        return {"activation": self.activation.__name__}

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.activation(inputs) 

    def backward(self, grad: Tensor) -> Tensor:
        return self.activation_prime(self.inputs) * gradient


class Relu(Activation):
    def __init__(self):
        super(Relu, self).__init__(relu, relu_prime)
