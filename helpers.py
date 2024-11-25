import os
import pprint
import numpy as np
import pandas as pd
import pulp as lp
import highspy
import time as time

from Examples.Example1 import example1
from Examples.Example2 import example2
from Examples.Example3 import example3
from Examples.Example4 import example4
from Examples.Example5 import example5
from Examples.Example6 import example6
from Examples.Example7 import example7
from Examples.Example8 import example8
from Examples.Example9 import example9
from Examples.Example10 import example10
from Examples.Example11 import example11
from Examples.Example12 import example12
from Examples.Example13 import example13
from Examples.Example14 import example14
from Examples.Example15 import example15
from Examples.Example16 import example16
from Examples.Example17 import example17
from Examples.Example18 import example18
from Examples.Example19 import example19
from Examples.Example20 import example20
from Examples.Example21 import example21
from Examples.Example22 import example22
from Examples.Example23 import example23
from Examples.Example24 import example24
from Examples.Example25 import example25
from Examples.Example26 import example26
from Examples.Example27 import example27


def get_problems():
    """
    Returns a dictionary of problem examples indexed by their corresponding number.

    This function provides a mapping of integers (from 1 to 25) to specific example problem instances.
    Each example corresponds to a function (e.g., `example1`, `example2`, etc.) that defines
    a particular problem in the context of the broader problem-solving framework (likely optimization or simplex problems).

    The returned dictionary allows for easy retrieval of any of the 25 problems by its index.

    Returns:
    --------
    dict:
        A dictionary where the keys are integers (1 to 25) and the values are function objects (example1, example2, ..., example25).

    Example:
    --------
    problems = get_problems()
    problem1 = problems[1]  # Retrieves example1 function
    problem5 = problems[5]  # Retrieves example5 function

    Usage:
    ------
    You can use this function to retrieve a specific example by calling `problems = get_problems()`
    and then accessing the desired problem with the corresponding integer key, such as `problems[1]`.
    """
    return {
        1: example1,
        2: example2,
        3: example3,
        4: example4,
        5: example5,
        6: example6,
        7: example7,
        8: example8,
        9: example9,
        10: example10,
        11: example11,
        12: example12,
        13: example13,
        14: example14,
        15: example15,
        16: example16,
        17: example17,
        18: example18,
        19: example19,
        20: example20,
        21: example21,
        22: example22,
        23: example23,
        24: example24,
        25: example25,
        26: example26,
        27: example27,
    }


def sherman_morrison_woodbury(Binv, Dim, U, V, TempB):
    """
    Applies the Sherman-Morrison-Woodbury formula to compute the inverse of (A + UCV^T).

    This function updates the inverse of matrix A (Binv) using the Sherman-Morrison-Woodbury formula,
    which is particularly useful when the matrix is being updated with a low-rank modification.

    Parameters:
    ----------
    Binv : numpy.ndarray
        The inverse of matrix B (n x n) that is being updated.
    Dim : int
        The dimension of the square identity matrix I used in the formula.
    U : numpy.ndarray
        The U matrix in the update (n x k).
    V : numpy.ndarray
        The V matrix in the update (n x k).

    Returns:
    -------
    numpy.ndarray
        The updated inverse matrix after applying the Sherman-Morrison-Woodbury formula.

    Example:
    --------
    updated_inverse = sherman_morrison_woodbury(Binv, Dim, U, V)
    """
    IVU = np.eye(Dim) + np.dot(np.dot(V, Binv), U)
    if np.linalg.det(IVU) == 0:
        return np.linalg.inv(TempB)

    inner_inv = np.linalg.inv(IVU)

    # Apply the full Sherman-Morrison-Woodbury formula
    term = np.dot(np.dot(np.dot(Binv, U), inner_inv), np.dot(V, Binv))

    # Return the updated inverse
    return Binv - term


def initiate_stats(A):
    """
    Initializes a statistics dictionary for tracking iterations, cuts, and other metrics during the simplex algorithm.

    Parameters:
    ----------
    A : numpy.ndarray
        The matrix for which statistics will be initialized, typically the constraint matrix of shape (m, m+n).

    Returns:
    -------
    dict
        A dictionary initialized with statistics such as iteration count, number of cuts, infeasibilities,
        and matrix dimensions.

    Example:
    --------
    stats = initiate_stats(A)
    """
    m, n = np.shape(A)
    n = n - m
    stats = {
        "n": n,
        "m": m,
        "Iteration": 0,
        "NbrIversions": 0,
        "NbrCuts": 0,
        "NbrInfeasibilities": 0,
        "NoPivots": 0,
        "NbrNodes": 0,
        "NbrNodes_i": 0,
        "NbrNodes_j": 0,
        "NbrNodes_k": 0,
        "Rank_0": 0,
        "Rank_1": 0,
        "Rank_2": 0,
        "Rank_3": 0,
        "Time": 0,
        "Message": "",
    }
    return stats


def CR_factorization(M):
    """
    Performs a rank-revealing factorization (Compact Representation) using Singular Value Decomposition (SVD).

    The function decomposes the matrix M into two smaller matrices U_hat and Vt (compact form of U and V),
    while filtering out small singular values below a certain tolerance to maintain numerical stability.

    Parameters:
    ----------
    M : numpy.ndarray
        The input matrix (m x n) to be factorized using SVD.

    Returns:
    -------
    U_hat : numpy.ndarray
        The left singular vectors corresponding to the significant singular values.
    Vt : numpy.ndarray
        The product of the significant singular values and the right singular vectors (S_hat * Vh).

    Example:
    --------
    U_hat, Vt = CR_factorization(M)
    """
    # Perform SVD
    U, s, Vh = np.linalg.svd(M, full_matrices=False)

    # Choose a tolerance level to determine rank
    tolerance = 1e-10

    # Determine the rank
    rank = np.sum(s > tolerance)

    # Construct the rank-revealing factorization
    U_hat = U[:, :rank]
    S_hat = np.diag(s[:rank])
    Vh_hat = Vh[:rank, :]

    Vt = np.dot(S_hat, Vh_hat)
    return U_hat, Vt


def create_directory_if_not_exists(directory):
    """
    Creates a directory if it does not already exist.

    Parameters:
    ----------
    directory : str
        The path of the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")


def create_mps(A, b, C, filename="model_maximize.mps"):
    """
    Creates an LP model in MPS format in the form:

        minimize Cx
        subject to Ax = b
        x >= 0

    Parameters:
    ----------
    A : np.ndarray
        The constraint matrix (m x n) for the LP problem.
    b : np.ndarray
        The right-hand side vector (m x 1) for the LP problem.
    C : np.ndarray
        The coefficients of the objective function (n x 1).
    filename : str, optional
        The name of the MPS file to save the model (default is 'model.mps').

    Returns:
    -------
    None
    """
    # Number of variables (n) and constraints (m)
    num_vars = A.shape[1]
    num_constraints = A.shape[0]

    # Create the LP model (assuming minimization problem)
    model = lp.LpProblem(name="LP_Problem", sense=lp.LpMinimize)

    # Define decision variables (x >= 0)
    x = [
        lp.LpVariable(f"x{i+1}", lowBound=0) for i in range(num_vars)
    ]  # Variables x1, x2, ..., xn

    # Add objective function (minimize Cx)
    model += lp.lpSum([C[i] * x[i] for i in range(num_vars)]), "Objective"

    # Add equality constraints (Ax = b)
    for i in range(num_constraints):
        model += (
            lp.lpSum([A[i][j] * x[j] for j in range(num_vars)]) == b[i]
        ), f"Constraint_{i+1}"

    # Write the model to an MPS file
    model.writeMPS(filename)

    print(f"MPS model saved to {filename}")


def create_problems_files(folder_name):
    create_directory_if_not_exists(folder_name)
    problems = get_problems()
    for i, p in problems.items():
        filename = f"{folder_name}/problem{i}.mps"
        A, b, C = p()
        C = -C.flatten()
        create_mps(A, b, C, filename=filename)


def solve_all_problems(folder_name):
    h = highspy.Highs()
    h.setOptionValue("solver", "simplex")
    h.setOptionValue("simplex_strategy", 4)
    h.setOptionValue("log_to_console", False)

    problems = get_problems()
    solutions = {}
    for i, _ in problems.items():
        ts = time.perf_counter()
        filename = f"{folder_name}/problem{i}.mps"
        status = h.readModel(filename)
        print("Reading model file ", filename, " returns a status of ", status)
        h.run()
        te = time.perf_counter()
        tspan = te - ts
        # solution = h.getSolution()
        # basis = h.getBasis()
        info = h.getInfo()
        model_status = h.getModelStatus()
        solution = {
            "problem": i,
            "status": h.modelStatusToString(model_status),
            "Optimal_objective": -info.objective_function_value,
            "Iterations": info.simplex_iteration_count,
            "Time": tspan,
        }
        solutions[i] = solution
        pprint.pprint(solution)
    df = pd.DataFrame.from_dict(solutions, orient="index")
    output_file = "./results/exact_solutions.xlsx"
    df.to_excel(output_file, index=False, sheet_name="Results")
    return solutions, info


if __name__ == "main":
    print("You're in helpers.py")
