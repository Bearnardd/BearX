from bearx.layers import RNN, Linear
from bearx.models import Sequential


def test_skeleton_function():
    model = Sequential()
    model.add(Linear(2, 1))
    model.add(RNN(1, 2))
    model.skeleton()
