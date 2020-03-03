import sys
sys.path.append("..")

import numpy as np
from bearx.losses import CrossEntropy

ce = CrossEntropy()

def test_feed_forward():
    predictions = np.array([[0.25, 0.25, 0.25, 0.25],
                            [0.01, 0.01, 0.01, 0.96]])

    targets = np.array([[0, 0, 0, 1],
                    [0, 0, 0, 1]])

    correct_ans = 0.7135581778200729

    x = ce.loss(predictions, targets)
    assert x == correct_ans

def test_backward():
    predictions = np.array([0.25, 0.63, 0.12])
    assert sum(predictions) == 1
    targets = np.array([0, 1, 0])

    x = ce.loss(predictions, targets)
    gradient = ce.backward
    print(x)


def test_model_with_softmax():
    from bearx.models import Sequential
    from bearx.layers import Linear, Softmax
    
    model = Sequential()
    model.add(Linear(3, 3, activation=Softmax()))
    model.compile()
    x = model.predict(np.array([1, 2, 3]))
    print(x)


if __name__ == "__main__":
    #test_backward()
    test_model_with_softmax()
