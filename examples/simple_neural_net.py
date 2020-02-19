import sys
sys.path.append("..")

import numpy as np

from bearx.layers import Tanh, Linear
from bearx.models import Sequential 
from bearx.losses import MSE
from bearx.optimizers import SGD
from bearx.tensor import Tensor

INPUT_SIZE = 2
HIDDEN_SIZE = 50
OUTPUT_SIZE = 1

X_train = np.array([np.array([i, i+1]) for i in range(101, 1024)])
y_train = np.array([np.array(sum(sample)) for sample in X_train])

# init sequential model
model = Sequential()
# add layers
model.add(Linear(INPUT_SIZE, HIDDEN_SIZE))
model.add(Tanh())
model.add(Linear(HIDDEN_SIZE, OUTPUT_SIZE))

# we can check structure of the model 
model.skeleton()


# we need to compile the model
model.compile()

# to train we just use train method


model.train(X_train, y_train, 10)
model.feed_forward([1, 2])

