import numpy as np
import sys
sys.path.append("..")

from tensor import Tensor 


class DataLoader(object):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self, inputs: Tensor, targets: Tensor):
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(inputs)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield batch_inputs, batch_targets
