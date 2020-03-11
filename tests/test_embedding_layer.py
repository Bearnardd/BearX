import sys
sys.path.append("..")

from bearx.layers import Embedding
from bearx.models import Sequential


model = Sequential()
model.add(Embedding(2, 2))
model.skeleton()
