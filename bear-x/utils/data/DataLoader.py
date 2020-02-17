import numpy as np
from typing import NamedTuple
import sys
sys.path.append("..")

from tensor import Tensor 

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataLoader(object):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor):
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(inputs)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)
            break