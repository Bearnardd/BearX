
class Optimizer:
    def step(self, model) -> None:
        raise NotImplementedError(
            "Not implemented in base class!"
        )


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, model) -> None:
        """
        Adjust layers params according
        to learning rate and gradient
        """
        for param, gradient in model.get_params_and_gradients():
            param -= self.lr * gradient
