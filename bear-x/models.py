"""
Let's start with simple Sequential model
"""
from layers import Layer 
from tensor import Tensor

class Model:
    def __init__(self):
        self.model = {}
        self.layer_idx = 0

    def add(self, layer: Layer):
        self.model[f"layer_{self.layer_idx}"] = layer
        self.layer_idx += 1

    def skeleton(self):
        """
        prints out model architecture
        """
        print(24 * ' ', "Model Summary")
        print(63 * '=')
        if self.layer_idx > 0:
            for name, layer in self.model.items():
                print(name.upper(), end="\n")
                print(layer.__getitem__(), end="\n")
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
        for layer in self.model.values():
            inputs = layer.feed_forward(inputs)
        return inputs

    def backward(self, gradient: Tensor) -> Tensor:
        for idx in reversed(range(self.layer_idx)):
            layer = self.model[f"layer_{idx}"]
            gradient = layer.back_propagation(gradient)
        return gradient

    def get_params_and_gradients(self):
        for layer in self.model.values():
            for name, param in layer.params.items():
                print(name)
                grad = layer.grads[name]
                yield param, grad

