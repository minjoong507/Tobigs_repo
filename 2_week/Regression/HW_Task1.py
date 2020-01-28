import numpy as np
from numpy.linalg import inv


def estimate_beta(x, y):
    new_x = inv(np.dot(x.T, x))
    new_y = np.dot(x.T, y)
    beta_hat = np.dot(new_x, new_y)

    return beta_hat


x = np.array([[1, 0, 1],
              [1, 2, 3],
              [1, 3, 8]
              ])

y = np.transpose(np.array([1, 3, 7]))
print(estimate_beta(x, y))
