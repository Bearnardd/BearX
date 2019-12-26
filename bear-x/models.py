"""
Let's start with simple Sequential model
"""
from layers import Layer 

class Model:
    def __init__(self, name):
        self.name = name
        self.model = {}
        self.idx = 0

    def add(self, layer: Layer):
        self.model[f"layer_{self.idx}"] = layer
        self.idx += 1

    def skeleton(self):
        for name, layer in self.model.items():
            print(name, end="\n")
            print(layer.__getitem__(), end="\n")

