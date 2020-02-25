import sys
sys.path.append("..")
from bearx.models import Sequential
from bearx.layers import Linear
import numpy as np


def test_predict_function():
    model = Sequential()
    model.add(Linear(2, 1))
    preds = model.predict(np.array([[1, 2, 4]]))
    print(preds)
    print(preds.shape)
    assert preds.shape == (1,3)

