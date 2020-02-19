import sys
sys.path.append("..")

from bearx.layers import Tanh, Linear
from bearx.models import Sequential 
from bearx.losses import MSE
from bearx.optimizers import SGD

INPUT_SIZE = 2
HIDDEN_SIZE = 40
OUTPUT_SIZE = 1

model = Sequential()
model.add(Linear(INPUT_SIZE, HIDDEN_SIZE))
model.add(Tanh())
model.add(Linear(HIDDEN_SIZE, OUTPUT_SIZE))

