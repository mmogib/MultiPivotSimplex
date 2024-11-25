# Example7:
import numpy as np


def example7():
    A = np.array(
        [
            [1, -2, 3, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 0, 0],
            [4, 9, 1, 4, 0, 0, 1, 0, 0],
            [2, 2, 1, 1, 0, 0, 0, 1, 0],
            [2, -1, 5, 0, 0, 0, 0, 0, 1],
        ]
    )
    b = np.array([[99], [40], [106], [60], [170]])
    C = np.array([[20, 12, 15, 6, 0, 0, 0, 0, 0]])
    return A, b, C
