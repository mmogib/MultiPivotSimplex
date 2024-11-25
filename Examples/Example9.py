import numpy as np


def example9():
    A = np.array(
        [
            [10, 11, 4, 16, 12, 1, 0, 0],
            [10, 7, 18, 16, 15, 0, 1, 0],
            [14, 17, 14, 10, 6, 0, 0, 1],
        ]
    )
    b = np.array([[20], [22], [30]])
    C = np.array([[3, 4, 19, 9, 3, 0, 0, 0]])
    return A, b, C
