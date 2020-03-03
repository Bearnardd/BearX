from typing import Dict
from bearx.tensor import Tensor
import numpy as np

# change it to *?
from bearx.activations import (
    relu, relu_prime, sigmoid, sigmoid_prime,
    tanh, tanh_prime
)

from bearx.gates import AddGate, MultiplyGate


class Layer:
    def __init__(self, **kwargs):
        """
        Layer base class
        """
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def __repr__(self):
        raise NotImplementedError(
            "Function not implemented in base class!"
        )

    def feed_forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError(
            "Function not implemented in base class!"
        )

    def backward(self, grad: Tensor) -> Tensor:
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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation=None,
        **kwargs
    ):
        super(Linear, self).__init__(**kwargs)
        # TODO: add activations
        # remove selfing features:w
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
            self.params["W"] = np.random.randn(
                self.in_features, self.out_features)
            self.params["b"] = np.random.randn(self.out_features)

    def __repr__(self, x: int = None):
        """
        Return information about layer
        """
        item = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "activation": (self.activation.__class__.__name__ if
                           self.activation is not None else "None")
        }
        return item

    def feed_forward(self, inputs: Tensor) -> Tensor:
        """
        element wise multiplication and addition
        :param: inputs: Tensor - input data
        :return: output matrix with activation function applied
        """
        # both methods give the same results
        # output = np.add(np.multiply(inputs, self.params["W"]) \
        # self.params["b"]))[0]
        self.inputs = inputs
        output = inputs @ self.params["W"] + self.params["b"]
        if self.activation:
            return self.activation.feed_forward(output)
        return output

    def backward(self, gradient: Tensor) -> Tensor:
        """
        self.grads["b"] = np.sum(gradient, axis=0)
        self.grads["W"] = np.multiply(self.inputs.T, gradient)
        return np.multiply(gradient, self.params["W"].T)
        """
        self.grads["b"] = np.sum(gradient, axis=0)
        self.grads["W"] = self.inputs.T @ gradient
        return gradient @ self.params["W"].T


class RNN(Layer):
    def __init__(
        self,
        rnn_units: int,
        in_features: int,
        out_features: int,
        activation=tanh,
        **kwargs
    ):
        super(RNN, self).__init__()

        allowed_kwargs = (
            "weight_initializer"
        )

        self.rnn_units = rnn_units
        self.mulGate = MultiplyGate()
        self.addGate = AddGate()

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError(f"Keyword not understood: {kwarg}")

        self.weight_initializer = kwargs.get("weight_initializer", None)

        if self.weight_initializer is None:
            self.params["U"] = np.random.rand(
                rnn_units, in_features
            )
            self.params["W"] = np.random.rand(
                rnn_units, rnn_units
            )
            self.params["V"] = np.random.rand(
                out_features, rnn_units
            )
        
        self.params["state"] = 0

    def __repr__(self):
        output = (f"# of units: {self.rnn_units}\n"
                  f"Shape of matrices: \n"
                  f"U: {self.params['U'].shape} "
                  f"W: {self.params['W'].shape} "
                  f"V: {self.params['V'].shape}"
                  f"State: {self.params['state'].shape}")
        return output

    def feed_forward(self, inputs: Tensor) -> Tensor:
        """
        self.inputs = inputs
        print(self.params["W_xh"] * inputs)
        print(np.dot(inputs, self.params["W_xh"].T))
        self.h = tanh(self.params["W_hh"] * self.h + self.params["W_xh"] * inputs)
        output = self.params["W_hy"] * self.h
        #print(output)#, self.h
        """

        self.inputs = inputs
        mulU = self.mulGate.forward(self.params["U"], inputs) 
        mulW = self.mulGate.forward(self.params["W"], self.params["state"])
        add = self.addGate.forward(mulU, mulW)
        self.params["state"]= tanh(add)
        output = self.mulGate.forward(self.params["V"], self.params["state"]) 
        return output
        
    def backward():
         


class Activation(Layer):
    def __init__(self, activation, activation_prime) -> None:
        super(Activation, self).__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def __repr__(self):
        return {"activation": self.activation.__name__}

    def feed_forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.activation(inputs)

    def backward(self, gradient: Tensor) -> Tensor:
        return self.activation_prime(self.inputs) * gradient


class Relu(Activation):
    def __init__(self):
        super(Relu, self).__init__(relu, relu_prime)


class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__(sigmoid, sigmoid_prime)


class Tanh(Activation):
    def __init__(self):
        super(Tanh, self).__init__(tanh, tanh_prime)
