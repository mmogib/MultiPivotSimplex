import numpy as np


def example1():
    A = np.array([[1, 3, 2, 1, 0, 0], [2, 2, 1, 0, 1, 0], [1, 1, 2, 0, 0, 1]])
    b = np.array([[4], [2], [3]])
    C = np.array([[2, 3, 2, 0, 0, 0]])
    return A, b, C
