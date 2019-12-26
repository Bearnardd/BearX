from typing import Dict
from tensor import Tensor
import numpy as np


class Layer:
    def __init__(self, **kwargs):
        """
        Layer base class
        """
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def back_prop(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    def __init__(self,
                 units: int,
                 in_features: int,
                 out_features: int,
                 output_size: int,
                 activation=None,
                 **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units
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
                self.out_features, 1) * 0.1

    def forward(self, inputs: Tensor) -> Tensor:
        """
        output = input * W + b
        """
        return inputs * self.params["W"] + self.params["b"]
