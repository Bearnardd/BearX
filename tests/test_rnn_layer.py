import sys
sys.path.append("../bearx")

import pytest
import numpy as np
from models import Sequential
from layers import RNN


def test_repr_function():
    """
    Testing __repr__ function of RNN
    """
    x = np.array([1, 1])
    rnn = RNN(1, 2)
    print(rnn)


@pytest.mark.parametrize("inputs", [np.array([[1, 1], [2, 2]]), np.array([[[2, 2], [2, 2]], [[1, 1], [1, 1]]])])
def test__call__function(inputs):
    rnn = RNN(2, 2)
    outputs = rnn(inputs)
    print(outputs)
    print(outputs.shape)


def test__call__function_shape_checking():
    with pytest.raises(ValueError) as e:
        x = np.array([1, 1])
        rnn = RNN(1, 2)
        assert rnn(x) == e


def test_model_backprop_with_rnn():
    inputs = np.array([[[1, 1], [2, 2], [3, 3], [1, 1]]])
    targets = np.array([[[2, 2], [3, 3], [4, 4], [2, 2]]])
    model = Sequential()
    model.add(RNN(2, input_shape=(3, 2)))
    model.skeleton()

    model.compile(
        batch_size=1
    )

    his = model.train(inputs, targets, 100, verbose=True)
    preds = model.predict(inputs)
    preds = preds.astype('int32') 
    print(preds)
