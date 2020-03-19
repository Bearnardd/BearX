import sys
sys.path.append("../bearx")

from layers import RNN
import numpy as np


def test_repr_function():
    x = np.array([1,1])
    rnn = RNN(1, 2, 1)
    print(rnn)

def test_weight_initialization_shapes():
    rnn = RNN(1, 2)
    assert rnn.params["U"].shape == (2, 1)
    assert rnn.params["W"].shape == (2, 2)
    assert rnn.params["V"].shape == (2, 2)


def test_forward_propagation():
    inputs = np.array([1, 2])
    rnn = RNN(1, 2)
    output = rnn(inputs)
    assert output.shape == (2,)


def test_forward_prediction_function():
    inputs = np.array([0, 1, 2])
    rnn = RNN(3, 3)
    RNNcells = rnn.forward(inputs)
    assert RNNcells.shape == (3,)

def test_calculate_loss_function():
    inputs = np.array([0, 1, 2])
    targets = np.array([2, 3, 4])
    rnn = RNN(1, 3)
    loss = rnn.calculate_loss(inputs, targets)
    print(loss)

def test_bptt_function():
    inputs = np.array([0, 1, 2])
    targets = np.array([0, 1, 2])
    rnn = RNN(3, 3)
    rnn.bptt(inputs, targets)
    



