from bearx.layers import Linear
from bearx.models import Sequential

import numpy as np


def test_adding_activations_to_layer_as_string():
    model = Sequential()
    model.add(Linear(1, 1, activation='relu'))
    output = model.feed_forward(np.array([-5]))
    assert output >= 0

