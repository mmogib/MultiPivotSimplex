import numpy as np


# Example4:
def example4():
    A = np.array(
        [
            [1, 2, 2, 2, 5, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 5, 1, 4, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 3, 2, 7, 1, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [2, 1, 7, 2, 6, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [3, 2, 1, 4, 3, 7, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 2, 5, 2, 5, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [3, 2, 1, 4, 2, 7, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [4, 3, 2, 8, 1, 6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [2, 1, 4, 2, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [4, 2, 1, 4, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    b = np.array([[9], [4], [5], [8], [7], [9], [6], [4], [3], [4]])
    C = np.array([[5, 4, 3, 5, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    return A, b, C
