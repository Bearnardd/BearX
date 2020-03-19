import sys
sys.path.append("../bearx")


from models import Sequential
from layers import RNN, Linear


def test_skeleton_function():
    model = Sequential()
    model.add(Linear(2, 1))
    model.add(RNN(1, 2))
    model.skeleton()
