import sys
sys.path.append("../bearx")

import numpy as np

from layers import Flatten, Embedding, Linear
from models import Sequential


def test_flatten_layer():
    f = Flatten()
    inputs = np.array([
        [1, 2],
        [3, 4]
    ])
    out = f(inputs)
    assert out.shape == (4,)


def test_flatten_layer_with_model():
    model = Sequential()
    model.add(Embedding(2, 2))
    model.add(Flatten())
    model.add(Linear(4, 1))

    inputs = np.array([0, 1])
    out = model.feed_forward(inputs)
    assert out.shape == (1,)
