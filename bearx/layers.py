from typing import Dict
from tensor import Tensor
import numpy as np

# change it to import initializers??
from activations import *
from gates import AddGate, MultiplyGate
from initializers import *
from backend import gather


addGate = AddGate()
mulGate = MultiplyGate()


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

    def __call__(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError(
            "Function not implemented in base class!"
        )

    def back_prop(self, grad: Tensor) -> Tensor:
        raise NotImplementedError(
            "Function not implemented in base class!"
        )


class Activation(Layer):
    def __init__(self, activation, activation_prime) -> None:
        super(Activation, self).__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def __repr__(self):
        return {"activation": self.activation.__name__}

    def forward(self, inputs: Tensor) -> Tensor:
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
        output = (
            f"in_features: {self.in_features}\n"
            f"out_features: {self.out_features}\n"
            f"activation: {self.activation.__class__.__name__ if self.activation is not None else 'None'}\n"
        )
        return output
        """
        item = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "activation": (self.activation.__class__.__name__ if
                           self.activation is not None else "None")
        }
        return item
        """

    def __call__(self, inputs: Tensor) -> Tensor:
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
            return self.activation.forward(output)
        return output

    def back_propagation(self, gradient: Tensor) -> Tensor:
        """
        self.grads["b"] = np.sum(gradient, axis=0)
        self.grads["W"] = np.multiply(self.inputs.T, gradient)
        return np.multiply(gradient, self.params["W"].T)
        """

        self.grads["b"] = np.sum(gradient, axis=0)
        self.grads["W"] = self.inputs.T @ gradient
        return gradient @ self.params["W"].T


class Embedding(Layer):
    """
    Turns indexes (integers) into dense vectors of given size

    # Example:

    ```python
        model = Sequential()
        model.add(Embedding(2, 2))

        |
        |
        v
        
        # the model will take as input


    @param: input_dim (int) : size o the vocabulary -> max int index + 1
    @param: output_dim (int) : dimension of the dense embedding
    @param: embeddings_initializer (Initializer) : initializer for embeddings matrix
    @param: input_length (int) : length of input sequences [if they are constant, else None]

    @__call__: return array of dense arrays -> convert indices to vectors of random numbers picked
    from given distribution

    ---------------------------------------------------
    | Can only be used as the first layer in a model! |
    ---------------------------------------------------
    """

    def __init__(self, input_dim: int, output_dim: int,
                 embeddings_initializer: Initializer = RandomUniform(-0.05, 0.05),
                 input_length: int = None,
                 **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.embeddings = self._build(embeddings_initializer)

    def _build(self, embeddings_initializer) -> Tensor:
        weight = embeddings_initializer(
            shape=(self.input_dim, self.output_dim))
        return weight

    def __repr__(self) -> str:
        return f"input_dim: {self.input_dim}, output_dim: {self.output_dim}"

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.dtype != "int32":
            inputs = inputs.astype('int32')
        out = gather(self.embeddings, inputs)
        return out


class RNN(Layer):
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            activation=Tanh(),
            **kwargs
        ):
        super(RNN, self).__init__()
        self.activation = activation

        allowed_kwargs = (
            "weight_initializer"
        )

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

        self.state = 0

    """

    def __repr__(self):
        output = (f"# of units: {self.rnn_units}\n"
                  f"Shape of matrices: \n"
                  f"W_xh: {self.params['U'].shape} "
                  f"W_hh: {self.params['W'].shape} "
                  f"W_hy: {self.params['V'].shape}")
        return output

    def forward(self, x, prev_state) -> Tensor:
        self.mulU = mulGate.forward(self.params["U"], x)
        self.mulW = mulGate.forward(self.params["W"], prev_state)
        self.add = addGate.forward(self.mulW, self.mulU)
        self.state = self.activation.forward(self.add)
        self.mulV = mulGate.forward(self.params["V"], self.state)
        return mulV

    def backward(self, x, prev_state, diff_s, dmulV):
        dV, dsv = mulGate.backward(self.params["V"], self.state, dmulV)
        dh = dsv + diff_s
        dadd = self.activation.backward(self.add, dh)
        dmulW, dmulU = addGate.backward(self.mulW, self.mulU, dadd)
        dW, dprev_S = mulGate.backward(self.params["W"], prev_state, dmulW)
        dU, dx = mulGate.backward(self.params["U"], x, dmulU)
        return (dprev_S, dU, dW, dV)
