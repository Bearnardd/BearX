#!/usr/bin/env python3
from bearx.tensor import Tensor
from bearx.layers import Relu, Sigmoid, Linear, Layer, Tanh
from bearx.losses import MSE, Loss
from bearx.models import Sequential 
from bearx.optimizers import Optimizer, SGD
from bearx.utils.data.DataLoader import DataLoader

import numpy as np
import pickle

from typing import List

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

def binary_encode(x: int) -> List[int]:
    """
    10 digit binary encoding of x
    """
    return [x >> i & 1 for i in range(10)]


if __name__ == "__main__":
    inputs = np.array([
        binary_encode(x)
        for x in range(101, 1024)
    ])

    targets = np.array([
        fizz_buzz_encode(x)
        for x in range(101, 1024)
    ])
    print(targets)

    model = Sequential()
    model.add(Linear(10 ,50))
    model.add(Tanh())
    model.add(Linear(50,4))
    model.skeleton()
    model.compile(lr=0.001)
    model.train(inputs, targets, 2000)
    #model.save_weights("test")
    #model.feed_forward([0,1,0,0,0,1,1,0,0,0])



    # [0100001001] -> [1000]
    # model.compile(batch_size=32, lr=0.001)
    # model.train(inputs, targets, 2000)
