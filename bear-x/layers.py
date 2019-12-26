from typing import Dict
from tensor import Tensor
import numpy as np
import timeit


class Layer:
    def __init__(self, **kwargs):
        """
        Layer base class
        """
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError(
            "Function not implemented in base class!"
        )

    def back_prop(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
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

    def forward(self, inputs: Tensor) -> Tensor:
        """
        element wise multiplication and addition
        :param: inputs: Tensor - input data
        :return: output matrix with activation function applied
        """
        # both methods give the same results
        # output = np.add(np.multiply(inputs, self.params["W"]), self.params["b"])[0]
        output = inputs @ self.params["W"] + self.params["b"]
        return self.activation(output)



