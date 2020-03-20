import sys
sys.path.append("../bearx")

import numpy as np
from layers import RNN


def test_repr_function():
    """
    Testing __repr__ function of RNN
    """
    x = np.array([1, 1])
    rnn = RNN(1, 2, 1)
    print(rnn)


def test_weight_initialization_shapes():
    """
    Testing default weight initializer for
    RNN which is bearx.initializers.RNNinit()
    """
    rnn = RNN(1, 2)
    assert rnn.params["U"].shape == (2, 1)
    assert rnn.params["W"].shape == (2, 2)
    assert rnn.params["V"].shape == (2, 2)


def test_output_shape_of_call_function():
    """
    Testing output shape while using
    feed forward rnn( __call__ function )
    which produces array of RNNCells
    """
    inputs = np.array([0, 1, 2])
    rnn = RNN(3, 3)
    RNNcells = rnn(inputs)
    assert RNNcells.shape == (3,)


def test_calculate_loss_function():
    """
    As described in Cross-Entropy loss docstring
    we clip prediction distribution to avoid
    log(0) or log(1) which means that even if
    are predictions are 100% right our loss 
    cannot be equal to 0.
    """
    inputs = np.array([0, 1, 2])
    targets = np.array([0, 1, 2])
    rnn = RNN(3, 3)
    loss = rnn.calculate_loss(inputs, targets)
    assert loss > 0


def test_bptt_function():
    inputs = np.array([0, 1, 2])
    targets = np.array([1, 2, 0])
    rnn = RNN(3, 3)
    rnn.bptt(inputs, targets)
