import numpy as np


# Example3:
def example3():
    A = np.array(
        [
            [1, 2, 1, 2, 1, 3, 1, 1, 0, 0],
            [1, 3, 4, 2, 1, 3, 1, 0, 1, 0],
            [3, 2, 2, 1, 5, 4, 1, 0, 0, 1],
        ]
    )
    b = np.array([[4], [4], [4]])
    C = np.array([[4, 3, 5, 3, 3, 4, 3, 0, 0, 0]])
    # return simplexiter(A, b, C, 3, 7)
    # return simplexiter(A, b, C)
    return A, b, C
