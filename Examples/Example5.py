import numpy as np

# Example5:


def example5():
    A = np.array(
        [[1, 1, 1, 3, 1, 1, 0, 0], [1, 4, 1, 3, 1, 0, 1, 0], [1, 2, 1, 4, 1, 0, 0, 1]]
    )
    b = np.array([[1], [2], [3]])
    C = np.array([[5, 3, 4, 2, 3, 0, 0, 0]])
    return A, b, C
