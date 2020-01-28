import numpy as np
from numpy.linalg import inv


def estimate_beta(x, y):
    beta_hat = []
    for x_list in x:
        x_list.append(1)

    return beta_hat


x = np.array([[1, 0, 1],
               [1, 2, 3],
               [1, 3, 8]])

y = np.transpose(np.array([1, 3, 7]))

for x_list in x:
    x_list = np.append(x_list, [1])
    print(x_list)
print(x)