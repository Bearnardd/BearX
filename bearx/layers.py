import numpy as np
import sys

# change it to import initializers??
from activations import *
from initializers import Initializer, RNNinit, RandomUniform, Zeros, Ones, RandomNormalDist
from backend import gather, Softmax
from tensor import Tensor

from datetime import datetime

from typing import Dict, Tuple


# TODO: add parameters function to Linear layer

class Layer:
    def __init__(self, **kwargs):
        """
        Layers base class. All layers avaiable
        in bearx inherit from it.
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

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError(
            "Function not implemented in base class!"
        )

    def parameters(self):
        """
        Return number of parameters in the layer
        """
        return 0


class Activation(Layer):
    def __init__(self, activation, activation_prime) -> None:
        super(Activation, self).__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def __repr__(self):
        return {"activation": self.activation.__name__}

    def __call__(self, inputs: Tensor) -> Tensor:
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


activation_functions = {
    "relu": Relu,
    "tanh": Tanh
}

weight_initializers = {
    "ones": Ones,
    "zeros": Zeros,
    "uniform": RandomUniform,
    "normal": RandomNormalDist,
    "rnn": RNNinit,
}


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

    def __call__(self, inputs: Tensor) -> Tensor:
        """
        element wise multiplication and addition
        :param: inputs: Tensor - inpt data
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
    ```

        |
        |
        v

        # the model will take as inpt matrix of shape (batch_size, inpt_length)
        # and convert it into dense vectors of given size (output_dim)

    @param: inpt_dim (int) : size o the vocabulary -> max int index + 1
    @param: output_dim (int) : dimension of the dense embedding
    @param: embeddings_initializer (Initializer) : initializer for embeddings matrix
    @param: inpt_length (int) : length of inpt sequences [if they are constant, else None]

    @__call__: return array of dense arrays -> convert indices to vectors of random numbers picked
    from given distribution

    # TODO: add example

    ---------------------------------------------------
    | Can only be used as the first layer in a model! |
    ---------------------------------------------------
    """

    def __init__(self, inpt_dim: int, output_dim: int,
                 embeddings_initializer: Initializer = RandomUniform(-0.05, 0.05),
                 inpt_length: int = None,
                 **kwargs):
        super(Embedding, self).__init__()

        self.inpt_dim = inpt_dim
        self.output_dim = output_dim
        self.inpt_length = inpt_length
        self.embeddings = self._build(embeddings_initializer)

    def _build(self, embeddings_initializer) -> Tensor:
        weight = embeddings_initializer(
            shape=(self.inpt_dim, self.output_dim))
        return weight

    def __repr__(self) -> str:
        return f"inpt_dim: {self.inpt_dim}, output_dim: {self.output_dim}"

    def display_embedding(self):
        print("+-----------+" + self.output_dim * 13 * "-" + "+")
        print("|   index   |" + self.output_dim * 5 * " " + "Embedding")
        print("+-----------+" + self.output_dim * 13 * "-" + "+")
        for idx, vector in enumerate(self.embeddings):
            print("|     " + str(idx) + "     |" +
                  self.output_dim * " " + str(vector))
        print("+-----------+" + self.output_dim * 13 * "-" + "+")

    def __call__(self, inputs: Tensor) -> Tensor:
        try:
            if inputs.dtype != "int32":
                inputs = inputs.astype('int32')
            out = gather(self.embeddings, inputs)
            return out
        except:
            raise AttributeError(
                "Inputs have to be type Tensor (np.array)!"
            )


class Flatten(Layer):
    """
    Flatten output of for example Embedding layer to be able to
    pass it through Linear layer (Dense/Fully Connected)

    # Example

    ```python
        model = Sequential()
        model.add(Embedding(2, 2))
        model.add(Flatten)
    ```
        |
        |
        v

    # as the output we got tensor with shape (4,)
    # so we flattened out, output from Embedding layer which was (2, 2)
    """

    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs.flatten()


"--------------------------------------------------------------------------------------------------------------"


class RNN(Layer):
    def __init__(self, input_size:  int, hidden_units: int, activation='tanh', bptt_trunc=4, weight_initializer='rnn'):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.activation = activation_functions[activation]()
        self.bptt_trunc = bptt_trunc
        self.weight_initializer = weight_initializers[weight_initializer](
            input_size, hidden_units)

        # init weights matrixes
        self.params["U"], self.params["W"], self.params["V"] = self.weight_initializer()
        self.grads["U"] = np.zeros_like(self.params["U"])
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["V"] = np.zeros_like(self.params["V"])

    def __repr__(self):
        output = (
            f"Shapes of weight matrices: \n"
            f"  U: {self.params['U'].shape}\n"
            f"  W: {self.params['W'].shape}\n"
            f"  V: {self.params['V'].shape}\n"
            f"W_initializer: {self.weight_initializer.__class__.__name__}\n"
            f"Number of params: {self.parameters()}")
        return output

    def parameters(self):
        return np.prod(self.params["U"].shape) + np.prod(self.params["W"].shape) + np.prod(self.params["V"].shape)

    def __call__(self, inputs: Tensor) -> Tensor:
        if len(inputs.shape) == 1:
            raise ValueError(
                f"Wrong input shape, should be (timesteps, input_dim) or (batch_size, timesteps, input_dim): got {inputs.shape}")
        if len(inputs.shape) == 2:
            inputs = np.expand_dims(inputs, axis=0)
        batch_size, timesteps, input_dim = inputs.shape
        print(batch_size, timesteps, input_dim)
