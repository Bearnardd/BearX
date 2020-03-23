import sys
sys.path.append("../bearx")

from layers import RNN
import numpy as np
import pytest


def test_repr_function():
    """
    Testing __repr__ function of RNN
    """
    x = np.array([1, 1])
    rnn = RNN(1, 2)
    print(rnn)

@pytest.mark.parametrize("inputs", [np.array([[1,1], [2,2]]), np.array([[[2,2], [2,2]], [[1,1], [1,1]]])])
def test__call__function(inputs):
    rnn = RNN(2, 2)
    rnn(inputs)

def test__call__function_shape_checking():
    with pytest.raises(ValueError) as e:
        x = np.array([1, 1])
        rnn = RNN(1, 2)
        assert rnn(x) == e

    
