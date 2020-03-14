import sys
sys.path.append("../bearx")

from layers import Embedding
from models import Sequential

import numpy as np

def test_embedding_layer_output_shape():
    e = Embedding(2, 4)
    out = e(np.array([0,1]))
    assert out.shape == (2, 4)


def test_embedding_layer_with_sequential_model():
    model = Sequential()
    model.add(Embedding(3, 6))
    word_dim = np.array([0, 1, 2])
    output = model.feed_forward(word_dim)
    assert output.shape == (3, 6)


def test_embedding_layer_with_ndim_2_input():
    model = Sequential()
    model.add(Embedding(3, 6))
    output = model.feed_forward(np.array([[0], [1], [2]]))
    assert output.shape == (3, 1, 6)


def test_embedding_layer_show_matrix_function():
    e = Embedding(5, 2)
    print("\n")
    e.display_embedding()
    e = Embedding(5, 3)
    e.display_embedding()
    e = Embedding(5, 4)
    e.display_embedding()



