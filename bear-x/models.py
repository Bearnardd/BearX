"""
Let's start with simple Sequential model
"""
from layers import Layer 
from tensor import Tensor

class Model:
    def __init__(self) -> None:
        self.layers = [] 

    def add(self, layer: Layer):
        self.layers.append(layer)

    def skeleton(self):
        """
        prints out model architecture
        """
        print(24 * ' ', "Model Summary")
        print(63 * '=')

        layer_idx = 0
        if self.layers:
            for layer in self.layers:
                name = layer.__class__.__name__
                print(f"{name.upper()} (layer_{layer_idx})", end="\n")
                print(layer.__getitem__(), end="\n")
                layer_idx += 1
        else:
            print(20 * " " + "Model has no layers yet!")
            print(17 * " " + "To add some use add() method")
        print(63 * '=')

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
        for layer in reversed(self.layers):
            gradient = layer.back_propagation(gradient)
        return gradient

    def get_params_and_gradients(self):
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
