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
        Derivative of MSE is just 2 * (predicted - target)
        """
        gradient = 2 * (predicted - target)
        return gradient


class BinaryCrossEntropy(Loss):
    def loss(self, predicted: Tensor, target: Tensor, epsilon=1e-12) -> float:
        """
        Computes cross entropy between target and prediced
        We compute cross entropy with the following formula:
        -∑p[x] * log(q[x]) where p is the true distribution
        and q is our predicted distribution. We can easily
        notice that p will be one hot encoded vector with
        0 everywhere except index of true label which will
        be 1 so in fact cross entropy will be log of our
        prediction that true label is in fact true label
        So our cross-entropy values will be between log(0)
        if we predicted 0% on true label and log(1) when
        our prediction was right. As You may know log(0) = infinity
        and log(1) = 0. So we want to clip our prediction with some
        epsilon to be sure that we avoid infinite loss which also
        implies that we can never get an excat zero-valued loss.

        @param: predicted (Tensor) -> tensor of our predictions

        @param: target (Tensor) -> tensor of true distribution

        @param: epsilon (float) -> small value we "add"(clip)
                    to our predictions to avoid infinite loss

        @return: cross entropy value (float) -> diffrnece in true
                    and predicted distributions
        """
        assert len(predicted) == len(target), "Lenghts of predicted and target have to be te same"
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        print(predicted.shape)
        ce = -np.sum(target * np.log(predicted))
        ce1 = - target * np.log(predicted) + (1 - target) * np.log(1 - predicted)
        print(ce, ce1)
        #return ce

    def backward(self, predicted: Tensor, target: int) -> Tensor:
        predicted = np.clip(pre)
        predicted[target] -= 1
        return predicted
