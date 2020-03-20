import numpy as np
import sys
sys.path.append("../bearX")

from preprocessing import getSentenceData
from layers import RNN

word_dim = 8000
hidden_dim = 100
X_train, y_train = getSentenceData('data/reddit-comments-2015-08.csv', word_dim)

print(X_train.shape)
print(y_train.shape)
np.random.seed(10)
rnn = RNN(word_dim, hidden_dim)

losses = rnn.train(X_train, y_train, learning_rate=0.005, nepoch=10, evaluate_loss_after=1)