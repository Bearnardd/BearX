#!/usr/bin/env python3
import sys
sys.path.append("..")

import numpy as np

from bearx.layers import Tanh, Linear, Relu
from bearx.models import Sequential 
from bearx.losses import MSE
from bearx.optimizers import SGD

import matplotlib.pyplot as plt


INPUT_SIZE = 3 
HIDDEN_SIZE = 50 
OUTPUT_SIZE = 1

inputs = np.array([[0,0,1],
                    [1,1,1],
                    [1,0,1],
                    [0,1,1]])

outputs = np.array([0,1,1,0])

# init sequential model
model = Sequential()
# add layers
model.add(Linear(INPUT_SIZE, OUTPUT_SIZE, activation=Relu()))
#model.add(Tanh())
#model.add(Linear(HIDDEN_SIZE, OUTPUT_SIZE))

# we can check structure of the model 
model.skeleton()

# we need to compile the model
model.compile(
    loss=MSE(),
    batch_size=1,
    optimizer=SGD(),
    lr=0.001,
)

# to train we just use train method
history = model.train(inputs, outputs, 2000, verbose=True)

# You can plot history of loss
plt.plot(history.history["loss"])
plt.show()

# evaluate model
preds = model.feed_forward(np.array([[0, 0, 1]]))

