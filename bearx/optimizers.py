import numpy as np


class Optimizer:
    def step(self, model) -> None:
        raise NotImplementedError(
            "Not implemented in base class!"
        )


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.5) -> None:
        self.lr = lr
        self.momentum = momentum
        self.w_update = None

    def step(self, model) -> None:
        """
        Adjust layers params according
        to learning rate and gradient
        """
        for param, gradient, w_update, name in model.get_params_and_gradients():
            #param -= self.lr * gradient
            update, w_updated = self.update(param, gradient, w_update)
            #print("NAAAAMe", name)
            #print("UU", w_updated)
            w_update = w_updated
            param -= update

    def update(self, w, grad_wrt_w, w_update):
        #if self.w_update is None:
            # np.shape to get shape for example from list
        #    self.w_update = np.zeros(np.shape(w))
        # equals 0 if momentum is 0
        #print("GRAD", grad_wrt_w)
        #print("MOM", self.momentum)
        w_update = self.momentum * w_update + (1 - self.momentum) * grad_wrt_w
        return self.lr * w_update, w_update
