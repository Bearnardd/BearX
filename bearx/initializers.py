import numpy as np


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
