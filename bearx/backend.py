import numpy as np


def gather(reference, indices):
    return reference[indices]


class Softmax:
    def __call__(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    def loss(self, x, y):
        print(x)
        print(y)
        probs = self(x)
        return -np.log(probs[y])

    def diff(self, x, y):
        probs = self(x)
        probs[y] -= 1.0
        return probs
