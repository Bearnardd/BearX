import numpy as np


class Optimizer:
    def step(self, model) -> None:
        raise NotImplementedError(
            "Not implemented in base class!"
        )


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0) -> None:
        self.lr = lr
        self.momentum = momentum

    def step(self, model) -> None:
        """
        Adjust layers params according
        to learning rate and gradient
        """
        if self.momentum != 0:
            for param, grad, w_update in model.get_params_and_gradients():
                print(param.shape, w_update.shape)
                w_updated = self.momentum * w_update  + (1 - self.momentum) * grad
                param -= self.lr * w_updated
                w_update += w_updated + w_update * 0
        else:
            for param, gradient in model.get_params_and_gradients():
                param -= self.lr * gradient