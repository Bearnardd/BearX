from bearx.tensor import Tensor

import numpy as np

# ignore numpy warning
import warnings
warnings.filterwarnings("ignore")


class Loss:
    def loss(self, predicted: Tensor, target: Tensor) -> float:
        raise NotImplementedError(
            "Not implemented in base class!"
        )

    def gradient(self, pred: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError(
            "Not implemented in base class!"
        )


class MSE(Loss):
    """
    Mean Squeared Error
    mse = sum((x - y) ** 2)
    """
    def loss(self, predicted: Tensor, target: Tensor) -> float:
        #assert type(predicted) == np.ndarray
        #assert type(target) == np.ndarray
        mse = np.sum((predicted - target) ** 2)
        return mse

    def gradient(self, predicted: Tensor, target: Tensor) -> Tensor:
        """
        derivative of mse is just 2 * (predicted - target)
        """
        gradient = 2 * (predicted - target)
        return gradient
