import sys
sys.path.append("..")

import numpy as np
from bearx.losses import CrossEntropy

ce = CrossEntropy()

predictions = np.array([[0.25, 0.25, 0.25, 0.25],
                        [0.01, 0.01, 0.01, 0.96]])

targets = np.array([[0, 0, 0, 1],
                [0, 0, 0, 1]])

correct_ans = 0.7135581778200729

x = ce.loss(predictions, targets)

if __name__ == "__main__":
    assert x == correct_ans
    print("Works!")
