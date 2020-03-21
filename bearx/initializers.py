import numpy as np
import six


class Initializer:
    def __call__(self, shape, dtype=None):
        raise NotImplementedError(
            "Not Implemented in base class"
        )

    def cfg(self):
        return {}


class Zeros(Initializer):
    def __call__(self, shape, dtype=None):
        return np.zeros(shape=shape, dtype=dtype)


class Ones(Initializer):
    def __call__(self, shape, dtype=None):
        return np.ones(shape=shape, dtype=dtype)


class Constant(Initializer):
    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=None):
        return np.full(fill_value=self.value, shape=shape, dtype=dtype)

    def cfg(self):
        return {"Value": self.value}


class RandomNormalDist(Initializer):
    def __init__(self, mean=0, std=1, seed=None):
        self.mean = mean
        self.std = std
        self.seed = seed

    def __call__(self, shape, dtype=None):
        if self.seed is not None:
            self.seed += 1
        if dtype is not None:
            dist = np.random.normal(self.mean, self.std, size=shape)
            return np.array(dist, dtype=dtype)
        return np.random.normal(self.mean, self.std, size=shape)

    def cfg(self):
        return {"mean": self.mean, "std": self.std, "seed": self.seed}


class RandomUniform(Initializer):
    def __init__(self, low, high, seed=None):
        self.low = low
        self.high = high
        self.seed = seed

    def __call__(self, shape, dtype=None):
        if self.seed is not None:
            self.seed += 1
        if dtype is not None:
            dist = np.random.uniform(self.low, self.high, size=shape)
            return np.array(dist, dtype=dtype)
        return np.random.uniform(self.low, self.high, size=shape)

    def cfg(self):
        return {"low": self.low, "high": self.high, "seed": self.seed}


class RNNinit(Initializer):
    """
    turns out that the best weight initialization is random in interval
    from -1/sqrt(n) to 1/sqrt(n) where n is the number of incoming connections
    from the previous layer
    """

    def __init__(self, word_dim: int, hidden_dim: int) -> None:
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim

    def __call__(self):
        U = np.random.uniform(-np.sqrt(1. / self.word_dim), np.sqrt(
            1. / self.word_dim), size=(self.hidden_dim, self.word_dim))
        W = np.random.uniform(-np.sqrt(1. / self.hidden_dim), np.sqrt(
            1. / self.hidden_dim), size=(self.hidden_dim, self.hidden_dim))
        V = np.random.uniform(-np.sqrt(1. / self.hidden_dim), np.sqrt(
            1. / self.hidden_dim), size=(self.word_dim, self.hidden_dim))
        return U, W, V

    def cfg(self):
        return {
            "U": self.U,
            "W": self.W,
            "V": self.V
        }


"""

AVAIABLE_INITIALIZERS = [Ones, Zeros,
                         Constant, RandomNormalDist, RandomUniform]


def get(identifier):
    if callable(identifier):
        return identifier
    elif isinstance(identifier, six.string_types):
        for initializer in AVAIABLE_INITIALIZERS:
            if identifier == initializer.__qualname__.lower():
                return initializer()
    elif isinstance(identifier, dict):
        if int
    else:
        raise ValueError(
            f"Could not interpret initializer identifier: {str(identifier)}"
        )
"""
