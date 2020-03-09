class Initializer:
    def __call__(self, shape, dtype=None):
        raise NotImplementedError(
            "Not Implemented in base class"
        )


class Zeros(Initializer):
    def __call__(self, shape, dtype=None):
        return np.zeros(shape=shape, dtype=dtype)
