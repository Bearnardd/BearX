from bearx.losses import CrossEntropy
from bearx.activations import softmax, softmax_prime

import numpy as np

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
    from models import Sequential
    from layers import Linear, Softmax

    inputs = np.array([[0.25, 0.63, 0.12]])
    targets = np.array([0, 1, 0])
    
    model = Sequential()
    model.add(Linear(3, 3, activation=Softmax()))
    predictions = model.feed_forward(inputs)
    loss = ce.loss(predictions, targets)
    for i in range(len(predictions)):
        gradient = ce.backward(predictions[i], targets[i])
        print("grad", gradient)

    #model.back_propagation()

def test_cross_entropy_loss_calc():
    predicted = np.array([0.1, 0.4, 0.5])
    targets = np.array([0, 0, 1])
    ce = CrossEntropy()
    ce.loss(predicted, targets)

    
