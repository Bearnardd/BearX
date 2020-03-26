from bearx.layers import Linear, Relu
from bearx.models import Sequential

import numpy as np


def test_adding_activations_to_layer_as_string():
    model = Sequential()
    model.add(Linear(1, 1, activation='relu'))
    output = model.feed_forward(np.array([-5]))
    assert output >= 0

def test_adding_activation_as_seperate_layer():
    model = Sequential()
    model.add(Linear(1, 1))
    model.add(Relu())
    model.skeleton()
    output = model.feed_forward(np.array([-5]))
    assert output >= 0

def test_parameters_function():
    s = Linear(2, 2)
    assert s.parameters() == 6
