"""
Let's start with simple Sequential model
"""
from layers import Layer 
from tensor import Tensor

class Model:
    def __init__(self):
        self.model = {}
        self.idx = 0

    def add(self, layer: Layer):
        self.model[f"layer_{self.idx}"] = layer
        self.idx += 1

    def skeleton(self):
        """
        prints out model architecture
        """
        print(24*' ', "Model Summary")
        print(63*'=')
        for name, layer in self.model.items():
            print(name.upper(), end="\n")
            print(layer.__getitem__(), end="\n")
        print(63*'=')
    def feed_forward(self, inputs: Tensor) -> Tensor:
        """
        Implementation of feed forward algorirthm
        We iterate over all Linear layers to get 
        transformed output
        """
        for layer in self.model.values():
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, gradient: Tensor) -> Tensor:
        pass

