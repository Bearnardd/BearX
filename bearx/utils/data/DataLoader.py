import numpy as np
from typing import NamedTuple
import sys
sys.path.append("..")

from bearx.tensor import Tensor 

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataLoader(object):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor):
        # TODO: not sure if it is gonna work in every case
        if len(targets.shape) == 1:
            targets = np.expand_dims(targets, axis=1)
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)
        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)
