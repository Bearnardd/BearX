import sys
sys.path.append("../bearx")

from layers import RNN
import numpy as np


def test_repr_function():
    x = np.array([1,1])
    rnn = RNN(1, 2, 1)
    rnn
