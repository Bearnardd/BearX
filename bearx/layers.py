import numpy as np
import sys

# change it to import initializers??
from bearx.activations import *
from bearx.initializers import Initializer, RNNinit, RandomUniform, Zeros, Ones, RandomNormalDist
from bearx.backend import gather, Softmax
from bearx.tensor import Tensor

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
        self.w_update: Dict[str, Tensor] = {}

    def __repr__(self):
        raise NotImplementedError(
            "Function not implemented in base class!"
        )

    def __call__(self, inputs: Tensor) -> Tensor:
        """
        Forward Propagation
        """
        raise NotImplementedError(
            "Function not implemented in base class!"
        )

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward Propagation
        """
        raise NotImplementedError(
            "Function not implemented in base class!"
        )

    def parameters(self):
        """
        Return number of parameters in the layer
        """
        return 0


class Activation(Layer):
    """
    Activations base class. All activation layers
    inherit from it
    """

    def __init__(self, activation, activation_prime) -> None:
        super(Activation, self).__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def __repr__(self):
        return f"activation: {self.activation.__name__}"

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
    fully-connected nn layer.

    Applies linear transformations on 
    input data. If specified can also 
    apply non-linear activation function

    Parameters:
    ----------
    in_features: int
        size of each input smaple 
    out_features: int
        size of each output sample 
    activation: Activation or str (None)
        activation function for this layer 
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation=None,
        **kwargs
    ):
        super(Linear, self).__init__(**kwargs)
        if activation != None:
            self.activation = activation_functions[activation]()
        else:
            self.activation = None
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
        Element wise multiplication and addition

        Parameters:
        -----------
        inputs: Tensor        
            Input data to feed through layer 

        Return:
        -------
        output: Tensor
        """
        self.inputs = inputs
        output = inputs @ self.params["W"] + self.params["b"]
        if self.activation:
            return self.activation(output)
        return output

    def backward(self, gradient: Tensor) -> Tensor:
        """
        Applies back propagation algorirthm,
        and saves weights gradients

        Parameters:
        -----------
        gradient: Tensor

        Return:
        -------
        gradient: Tensor

        self.grads["b"] = np.sum(gradient, axis=0)
        self.grads["W"] = np.multiply(self.inputs.T, gradient)
        return np.multiply(gradient, self.params["W"].T)
        """
        self.grads["b"] = np.sum(gradient, axis=0)
        self.grads["W"] = self.inputs.T @ gradient
        return gradient @ self.params["W"].T

    def parameters(self):
        return np.prod(self.params["W"].shape) + np.prod(self.params["b"].shape)


class Embedding(Layer):
    """
    Turns indexes (integers) into dense vectors of given size
    ---------------------------------------------------
    | Can only be used as the first layer in a model! |
    ---------------------------------------------------

    Parameters:
    -----------
    input_dim: int
        size o the vocabulary -> max int index + 1
    output_dim: int
        dimension of the dense embedding
    embeddings_initializer: Initializer
        initializer for embeddings matrix
    inpt_length: int
        length of input sequences [if they are constant, else None]

    # Example:

    ```python
        model = Sequential()
        model.add(Embedding(2, 2))
    ```
    |
    |
    v
    The model will take as inpt matrix of shape (batch_size, input_length)
    and convert it into dense vectors of given size (output_dim)
    """

    def __init__(self, inpt_dim: int, output_dim: int,
                 embeddings_initializer: Initializer = RandomUniform(-0.05, 0.05),
                 inpt_length: int = None,
                 **kwargs):
        super(Embedding, self).__init__()

        self.input_dim = inpt_dim
        self.output_dim = output_dim
        self.input_length = inpt_length
        self.embeddings = self._build(embeddings_initializer)

    def _build(self, embeddings_initializer) -> Tensor:
        embeddings = embeddings_initializer(
            shape=(self.input_dim, self.output_dim))
        return embeddings

    def __repr__(self) -> str:
        return f"inpt_dim: {self.input_dim}, output_dim: {self.output_dim}"

    def display_embedding(self):
        """
        Helper function to vizualize what is going on
        inside embeddings
        """
        print("+-----------+" + self.output_dim * 13 * "-" + "+")
        print("|   index   |" + self.output_dim * 5 * " " + "Embedding")
        print("+-----------+" + self.output_dim * 13 * "-" + "+")
        for idx, vector in enumerate(self.embeddings):
            print("|     " + str(idx) + "     |" +
                  self.output_dim * " " + str(vector))
        print("+-----------+" + self.output_dim * 13 * "-" + "+")

    def __call__(self, inputs: Tensor) -> Tensor:
        """
        Convert indices to vectors of random numbers  
        generated from given distribution (RandomUniform)

        Parameters:
        -----------
        inputs: Tensor:
            tensor of indices

        Return:
        _______
        Tensor of dense tensors
        """
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
    Flatten output of the previous layer 


    Return:
    -------
    output: Tensor
        Flattened output of the previous layer

    # Example

    ```python
        model = Sequential()
        model.add(Embedding(2, 2))
        model.add(Flatten)
    ```
    |
    |
    v
    We flatten output from embedding layer of shape(2, 2),
    so we end up with a tensor of shape(4,) [2 * 2]
    """

    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs.flatten()


"--------------------------------------------------------------------------------------------------------------"


class RNN(Layer):
    """
    The 'vanilla' rnn layer

    Parameters:
    -----------
    hidden_units: int
        Number of hidden states(cells) in the layer
    input_shape: Tuple (None)
        Shape of input data (timesteps, input_dim)        
    activation: string
        Activation function which will be applied to the output of the layer
    bptt_trunc: 
        Decides how many time steps the gradient should be propagated
        backwards through internal states given the loss gradient for
        time step t
    weight_initializer: string ('rnn')
        Type of weights initialization 
    """

    def __init__(self, hidden_units: int, input_shape: Tuple = None, activation='tanh', bptt_trunc=4, weight_initializer='rnn'):
        super(RNN, self).__init__()
        timesteps, input_dim = input_shape
        self.hidden_units = hidden_units
        self.activation = activation_functions[activation]()
        self.bptt_trunc = bptt_trunc
        self.weight_initializer = weight_initializers[weight_initializer](
            input_dim, hidden_units)

        # init weights matrixes
        self.params["U"], self.params["W"], self.params["V"] = self.weight_initializer()
        self.w_update["U"] = np.zeros_like(self.params["U"])
        self.w_update["W"] = np.zeros_like(self.params["W"])
        self.w_update["V"] = np.zeros_like(self.params["V"])

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

        self.state_input = np.zeros((batch_size, timesteps, self.hidden_units))
        # states are just internal states of rnn cells
        self.states = np.zeros((batch_size, timesteps+1, self.hidden_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))

        self.layer_input = inputs

        self.states[:, -1] = np.zeros((batch_size, self.hidden_units))
        for t in range(timesteps):
            # input for state t is current input and output of previous hidden state
            self.state_input[:, t] = inputs[:, t].dot(
                self.params["U"].T) + self.states[:, t-1].dot(self.params["W"].T)
            # apply activation function
            self.states[:, t] = self.activation(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.params["V"].T)

        return self.outputs

    def backward(self, accum_gradient: Tensor) -> Tensor:
        _, timesteps, _ = accum_gradient.shape

        grad_U = np.zeros_like(self.params["U"])
        grad_W = np.zeros_like(self.params["W"])
        grad_V = np.zeros_like(self.params["V"])

        # will be passed on to the previous layer in the network
        grad_next = np.zeros_like(accum_gradient)

        # bptt
        for t in reversed(range(timesteps)):
            grad_V += accum_gradient[:, t].T.dot(self.states[:, t])
            # w.r.t state input
            grad_wrt_state = accum_gradient[:, t].dot(
                self.params["V"]) * self.activation.backward(self.state_input[:, t])
            # w.r.t layer input
            grad_next[:, t] = grad_wrt_state.dot(self.params["U"])
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_ - 1])
                # w.r.t previous state
                grad_wrt_state = grad_wrt_state.dot(
                    self.params["W"]) * self.activation.backward(self.state_input[:, t-1])

        # update weights
        self.grads["U"] = grad_U
        self.grads["W"] = grad_W
        self.grads["V"] = grad_V

        return grad_next
