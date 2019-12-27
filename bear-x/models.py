"""
Let's start with simple Sequential model
"""
from layers import Layer 

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
        for name, layer in self.model.items():
            print(name, end="\n")
            print(layer.__getitem__(), end="\n")
    
    def feed_forward(self, x):
        """
        Implementation of feed forward algorirthm
        We iterate over all Linear layers to get 
        transformed output
        """
        for layer in self.model.values():
            layer_type = layer.__class__.__name__
            if layer_type == "Linear":
                x = layer.forward(x)
        return x

