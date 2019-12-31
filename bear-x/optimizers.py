from models import Model


class Optimizer:
    def step(self, model: Model) -> None:
        raise NotImplementedError(
            "Not implemented in base class!"
        )


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr

    def step(self, model: Model) -> None:
        for param, grad in model.get_params_and_gradients():
            param -= self.lr * grad
