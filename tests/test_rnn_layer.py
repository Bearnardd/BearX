import sys
sys.path.append("..")

from bearx.layers import RNN
import numpy as np

x = np.array([1,1])
rnn = RNN(1, 2, 2)
rnn
rnn.feed_forward(x)