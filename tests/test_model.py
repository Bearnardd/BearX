from bearx.models import Sequential
from bearx.layers import Linear

import numpy as np


def test_predict_tensor_expand_dim():
    model = Sequential()
    model.add(Linear(2, 1))
    preds = model.predict(np.array([1, 2]))
    assert preds.shape == (1, 1)


def test_predict_function_output_shape():
    model = Sequential()
    model.add(Linear(5, 3))
    preds = model.predict(np.array([[1, 2, 4, 6, 7]]))
    assert preds.shape == (1, 3)
