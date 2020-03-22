import numpy as np
import sys

# change it to import initializers??
from activations import *
from gates import AddGate, MultiplyGate
from initializers import Initializer, RNNinit, RandomUniform
from backend import gather, Softmax
from tensor import Tensor

from datetime import datetime

from typing import Dict, Tuple


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

addGate = AddGate()
mulGate = MultiplyGate()


class RNNCell:
    def __init__(self, activation=Tanh()):
        self.activation = activation
        self.state = 0

    def __call__(self, x, prev_state, U, W, V) -> None:
        self.mulU = mulGate.forward(U, x)
        self.mulW = mulGate.forward(W, prev_state)
        self.add = addGate.forward(self.mulW, self.mulU)
        self.state = self.activation(self.add)
        self.mulV = mulGate.forward(V, self.state)

    def backward(self, x, prev_s, U, W, V, diff_s, dmulV) -> Tuple:
        self(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(V, self.state, dmulV)
        ds = dsv + diff_s
        dadd = self.activation.backward(self.add)
        dmulw, dmulu = addGate.backward(self.mulW, self.mulU, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return dprev_s, dU, dW, dV


class RNN(Layer):
    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        return_sequences: bool = False,
        return_state: bool = False,
        activation=Tanh(),
        **kwargs
    ):
        super(RNN, self).__init__()
        # TODO: think if I need those
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.activation = activation
        self.bptt_truncate = 4
        self.weight_initializer = RNNinit(input_size, hidden_dim)

        # TODO: add arguments
        allowed_kwargs = [
            "weight_initializer"
        ]

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError(f"Keyword not understood: {kwarg}")

        # self.weight_initializer = kwargs.get(
        #    "weight_initializer", RNNinit(input_size, hidden_dim))

        self.params["U"], self.params["W"], self.params["V"] = self.weight_initializer()
        self.grads["U"] = np.zeros(self.params["U"].shape)
        self.grads["W"] = np.zeros(self.params["W"].shape)
        self.grads["V"] = np.zeros(self.params["V"].shape)

        def __repr__(self):
            output = (
                f"Shapes of weight matrices: \n"
                f"  U: {self.params['U'].shape}\n"
                f"  W: {self.params['W'].shape}\n"
                f"  V: {self.params['V'].shape}\n"
                f"  W_initializer: {self.weight_initializer.__class__.__name__}")
            return output

    def __call__(self, inputs: Tensor) -> Tensor:
        # number of time steps
        T = len(inputs)
        rnn_cells = []
        prev_state = np.zeros(self.hidden_dim)
        for t in range(T):
            layer = RNNCell(self.activation)
            inpt = np.zeros(self.input_size)
            inpt[inputs[t]] = 1
            layer(inpt, prev_state,
                  self.params["U"], self.params["W"], self.params["V"])
            if t == T - 1:
                print("U", self.params["U"])
            prev_state = layer.state
            rnn_cells.append(layer)
        return np.asarray(rnn_cells)

    def predict(self, x):
        output = Softmax()
        layers = self(x)
        return [np.argmax(output(layer.mulV)) for layer in layers]

    def bptt(self, x, y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self(x)

        T = len(layers)
        prev_s_t = np.zeros(self.hidden_dim)
        diff_s = np.zeros(self.hidden_dim)
        for t in range(0, T):
            dmulV = output.diff(layers[t].mulV, y[t])
            inpt = np.zeros(self.input_size)
            inpt[x[t]] = 1
            dprev_s, dU_t, dW_t, dV_t = layers[t].backward(
                inpt, prev_s_t, self.params["U"], self.params["W"], self.params["V"], diff_s, dmulV)
            prev_s_t = layers[t].state
            dmulV = np.zeros(self.input_size)
            for i in range(t-1, max(-1, t-self.bptt_truncate-1), -1):
                inpt = np.zeros(self.input_size)
                inpt[x[i]] = 1
                prev_s_i = np.zeros(
                    self.hidden_dim) if i == 0 else layers[i-1].state
                dprev_s, dU_i, dW_i, dV_i = layers[i].backward(
                    inpt, prev_s_i, self.params["U"], self.params["W"], self.params["V"], dprev_s, dmulV)
                dU_t += dU_i
                dW_t += dW_i
            self.grads["U"] += dU_t
            self.grads["W"] += dW_t
            self.grads["V"] += dV_t
            #dV += dV_t
            #dU += dU_t
            #dW += dW_t

    def sgd_step(self, x, y, learning_rate):
        self.bptt(x, y)
        self.params["U"] -= learning_rate * self.grads["U"]
        self.params["W"] -= learning_rate * self.grads["W"]
        self.params["V"] -= learning_rate * self.grads["V"]

    def calculate_loss(self, x, y):
        assert len(x) == len(y), "Lengths of x and y are not the same!"
        output = Softmax()
        layers = self(x)
        loss = 0.0
        for i, layer in enumerate(layers):
            loss += output.loss(layer.mulV, y[i])
        return loss / float(len(y))

    def calculate_total_loss(self, X, Y):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i], Y[i])
        return loss / float(len(Y))

    def train(self, X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        num_examples_seen = 0
        losses = []
        for epoch in range(nepoch):
            if epoch % evaluate_loss_after == 0:
                loss = self.calculate_total_loss(X, Y)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" %
                      (time, num_examples_seen, epoch, loss))
                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
            # For each training example...
            for i in range(len(Y)):
                self.sgd_step(X[i], Y[i], learning_rate)
                num_examples_seen += 1
        return losses
