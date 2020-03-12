import sys
sys.path.append("../bearx")

from layers import Embedding
from models import Sequential

import numpy as np


#model = Sequential()
#model.add(Embedding(2, 2))
#model.skeleton()
e = Embedding(2, 2)
x = e(np.array([5, 2, 5]))
