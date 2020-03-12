import sys
sys.path.append("../bearx")

from layers import Embedding
from models import Sequential

import numpy as np

def test_embedding_layer_output_shape():
    e = Embedding(2, 4)
    out  = e(np.array([0,1]))
    assert out.shape == (2, 4)
