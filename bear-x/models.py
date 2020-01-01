"""
Let's start with simple Sequential model
"""
from layers import Layer 
from tensor import Tensor

class Model:
    def __init__(self):
        self.layers = [] 

    def add(self, layer: Layer):
        self.layers.extend(layer)

    def skeleton(self):
        """
        prints out model architecture

        print(24 * ' ', "Model Summary")
        print(63 * '=')
        if layers:
            for layer in self.layers:
                print(name.upper(), end="\n")
                print(layer.__getitem__(), end="\n")
        else:
            print(20 * " " + "Model has no layers yet!")
            print(17 * " " + "To add some use add() method")
        print(63 * '=')
        """

    def feed_forward(self, inputs: Tensor) -> Tensor:
        """
        Implementation of feed forward algorirthm
        We iterate over all Linear layers to get 
        transformed output
        """
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
        return inputs

    def backward(self, gradient: Tensor) -> Tensor:
        for layers in reversed(self.layers):
            gradient = layer.back_propagation(gradient)
        return gradient

    def get_params_and_gradients(self):
        for layer in self.layers.values():
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
