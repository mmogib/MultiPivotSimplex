import itertools as itr
import numpy as np
from numpy import linalg as LA  # import linear algebra
from numpy.linalg import inv
import time as time

from helpers import CR_factorization, initiate_stats, sherman_morrison_woodbury

# from inner_loop import process_all_indices


def multipivot_simplex(
    A, b, C, eps=1e-8, maxiters=1000, Cut=True, Order=True, Pivots=2
):
    """
    Performs the Double Simplex algorithm to solve a linear programming problem.

    The objective is to maximize the linear function:
        Maximize: CX
        Subject to: AX = b and X >= 0

    Parameters:
    ----------
    A : numpy.ndarray
        The coefficient matrix of shape (m, n+m), where `m` is the number of constraints
        and `n+m` is the number of variables (including slack variables).
    b : numpy.ndarray
        The right-hand side vector of shape (m, 1) representing the constraints.
    C : numpy.ndarray
        The objective function coefficients, a row vector of shape (1, n+m).
    maxiters : int, optional
        The maximum number of iterations to perform (default is 1000).
    Cut : bool, optional
        Whether to apply cuts in the optimization process (default is True).
    Order : bool, optional
        Whether to reorder the Q matrix based on maximum possible improvement (default is True).
    Pivots : int, optional
        The number of pivots to explore when optimizing pairs of entering variables (default is 2).

    Returns:
    -------
    Z : float
        The optimal value of the objective function.
    X : numpy.ndarray
        The optimal solution vector of shape (n+m, 1).
    RC : numpy.ndarray
        The reduced costs vector of shape (1, n+m) corresponding to the solution.
    stats : dict
        A dictionary containing performance statistics such as iteration count, number of inversions,
        number of cuts, and total execution time.

    Notes:
    -----
    - The algorithm iterates until the reduced costs are non-positive or the maximum number of iterations is reached.
    - If no solution is found, the algorithm may return messages indicating whether the problem is unbounded
      or if the maximum number of iterations has been exceeded.

    Example:
    --------
    Z, X, RC, stats = multipivot_simplex(A, b, C, maxiters=500)
    """

    # intialization
    m, n = np.shape(A)
    n = n - m
    Z = 0
    X = np.zeros((n + m, 1))
    XB = np.zeros((m, 1))
    CB = np.zeros((1, m))
    MaxCB = np.zeros((1, m))
    XN = np.zeros((n, 1))
    CN = np.zeros((1, n))
    RC = np.zeros((1, n + m))
    Basis = [n + i for i in range(0, m)]
    B = A[:, Basis]
    NB = A[:, 0:n]
    CB[0, :] = C[0, Basis]
    CN[0, :] = C[0, 0:n]

    MaxB = np.copy(B)
    TempBi = np.zeros((m, m))
    TempBj = np.zeros((m, m))

    MaxBasis = Basis.copy()
    Bik = np.zeros((m, m))
    Bjk = np.zeros((m, m))
    Bk = np.zeros((m, m))

    Positive_RC_Indexes = [
        i for i in range(0, n + m)
    ]  # Lists all indexes of variables with strictly positive reduced cost
    Leaving_Rows_Indexes = [
        i for i in range(0, n + m)
    ]  # Lists all indexes of possible leaving rows

    Index_Enter: int = 0
    Index_Leave: int = 0

    # stats
    stats = initiate_stats(A)
    stats["NoPivots"] = Pivots
    Z = np.dot(CB, b)

    Binv = inv(MaxB)
    stats["NbrIversions"] += 1
    X = np.dot(Binv, b)
    Z = np.dot(CB, X)
    RC = C - np.dot(CB, np.dot(Binv, A))

    MaxRC = np.max(RC[0, :])
    if MaxRC <= 0:
        stats["Message"] = "No need to Optimize"
        return Z[0][0], X, RC, stats

    tic = time.perf_counter()
    while MaxRC > eps:  # and MaxVal>=eps and Degen <10
        if stats["Iteration"] > maxiters:
            stats["Message"] = f"Maximum Number of iterations ({maxiters}) reached"
            stats["Time"] = time.perf_counter() - tic
            return Z[0][0], X, RC, stats
        stats["Iteration"] = stats["Iteration"] + 1

        positive_rcs = np.where(RC[0, :] > eps)[0]
        Q = len(positive_rcs)
        QMatrix = np.zeros((Q, Q + 1))
        Positive_RC_Indexes = list(positive_rcs) + [-1 for i in range(Q + 1, n + m)]
        Leaving_Rows_Indexes = [*itr.repeat(-1, n + m)]

        Mat = np.dot(Binv, A)

        # ------------------------------------------------------------------------------------------
        Ratios = np.zeros((1, Q))
        # This loop checks which variable should leave the basis if our variable Index_Enter enters the basis
        for i in range(0, Q):
            Index_Enter = Positive_RC_Indexes[
                i
            ]  # Index of the variable that is under investigation to enter the basis

            Index_Leave = -1  # Index of the variable that could leave the basis
            MinVal = 10000000
            Binvb = np.dot(Binv, b)

            for j in range(0, m):
                val = Mat[j, Index_Enter]
                if val != 0:
                    bratio = Binvb[j, 0] / val
                    if bratio >= 0:
                        if MinVal > bratio:
                            # Tells us which variable should leave if Index_enter enters the basis
                            Index_Leave = j
                            MinVal = bratio
                            Ratios[0, i] = MinVal
            if Index_Leave == -1:
                stats["Message"] = "Problem Unbounded"
                stats["Time"] = time.perf_counter() - tic
                return Z[0][0], X, RC, stats
            Leaving_Rows_Indexes[i] = Index_Leave

        # Fill in the Q matrix
        MaxVal = 0
        for i in range(0, Q):
            QMatrix[i, i] = 1
            Index_Enter_i = Positive_RC_Indexes[i]
            QMatrix[i, Q] = float(Ratios[0, i] * RC[0, Index_Enter_i])

            for j in range(i + 1, Q):
                if Leaving_Rows_Indexes[i] != Leaving_Rows_Indexes[j]:
                    QMatrix[i, j] = 1
                    Index_Enter_j = Positive_RC_Indexes[j]
                    QMatrix[i, Q] = QMatrix[i, Q] + float(
                        Ratios[0, j] * RC[0, Index_Enter_j]
                    )

            MaxVal = max(MaxVal, QMatrix[i, Q])

        # Reorder QMatrix according to max possible improvement total combination
        if Order:
            for i in range(0, Q):
                for j in range(i + 1, Q):
                    if QMatrix[i, Q] < QMatrix[j, Q]:
                        tempi = Positive_RC_Indexes[i]
                        Positive_RC_Indexes[i] = Positive_RC_Indexes[j]
                        Positive_RC_Indexes[j] = tempi
                        # for k in range(0, Q + 1):
                        tempQ = QMatrix[i, :].copy()
                        QMatrix[i, :] = QMatrix[j, :].copy()
                        QMatrix[j, :] = tempQ.copy()

        # Now, we need to know how much improvement a combination (i,j) of entering variables can bring
        # Z = np.dot(CB, np.dot(Binv, b))
        DeltaZ = 0
        MaxDeltaZ = 0
        ################################################################################################
        for i in range(0, Q):
            ICutCondition = QMatrix[i, Q] >= MaxDeltaZ + eps if Cut else True
            if ICutCondition:
                stats["NbrNodes"] += 1
                stats["NbrNodes_i"] += 1
                TempBasis_i = Basis.copy()
                CheckInBasis_i = Basis.copy()
                DeltaZ = 0
                # find the last positive RC
                RCi = RC[0, Positive_RC_Indexes[i]]

                test1 = Leaving_Rows_Indexes[i]
                TempBasis_i[test1] = Positive_RC_Indexes[
                    i
                ]  # Insert variable with rc>0 in basis
                CheckInBasis_i[test1] = Positive_RC_Indexes[i]

                xi_star = Ratios[0, i]
                DeltaZ = xi_star * RCi
                TempBasis_i = [int(indx) for indx in TempBasis_i]
                if DeltaZ >= MaxDeltaZ + eps:
                    MaxDeltaZ = DeltaZ
                    MaxBasis = TempBasis_i.copy()
                    MaxB = A[:, MaxBasis]
                    MaxCB[0, :] = C[0, MaxBasis]

                TempBi = A[:, TempBasis_i]
                if np.linalg.det(TempBi) == 0:
                    continue
                DeltaB = TempBi - B
                Dim = LA.matrix_rank(DeltaB)
                stats[f"Rank_{Dim}"] = stats[f"Rank_{Dim}"] + 1
                U, V = CR_factorization(DeltaB)

                CheckBinv = sherman_morrison_woodbury(Binv, Dim, U, V, TempBi)
                TempBinv_i = CheckBinv.copy()
                stats["NbrIversions"] += (Dim * Dim) / (m * m)

                # Now we look for best pairs of entering variables
                ##############################################################################################
                # BB = B.copy()

                for j in range(0, Q):
                    if i != j:
                        if QMatrix[i, j] == 1:
                            QMatrix[j, i] = 0
                            JCutCondition = (
                                QMatrix[i, Q] >= MaxDeltaZ + eps if Cut else True
                            )
                            if JCutCondition:
                                stats["NbrNodes"] += 1
                                stats["NbrNodes_j"] += 1

                                TempBasis_j = TempBasis_i.copy()
                                CheckInBasis_j = CheckInBasis_i.copy()
                                test2 = Leaving_Rows_Indexes[j]
                                TempBasis_j[test2] = Positive_RC_Indexes[
                                    j
                                ]  # Insert variable with rc>0 in basis
                                CheckInBasis_j[test2] = Positive_RC_Indexes[j]

                                Feasible = True
                                TempBj = A[:, CheckInBasis_j]
                                if np.linalg.det(TempBj) == 0:
                                    # print(i, "huh", j, "")
                                    continue
                                TempB = TempBj.copy()
                                DeltaB = TempBj - TempBi
                                # DeltaB = TempBj - BB

                                Dim = LA.matrix_rank(DeltaB)
                                stats[f"Rank_{Dim}"] = stats[f"Rank_{Dim}"] + 1
                                U, V = CR_factorization(DeltaB)

                                CheckBinv = sherman_morrison_woodbury(
                                    TempBinv_i, Dim, U, V, TempBj
                                )
                                CheckInMaxX = np.dot(CheckBinv, b)
                                stats["NbrIversions"] += (Dim * Dim) / (m * m)
                                Feasible = np.all(CheckInMaxX[:, 0] >= 0)
                                if Feasible == True:
                                    Basis = CheckInBasis_j.copy()
                                    B = A[:, CheckInBasis_j]
                                    CB[0, :] = C[0, CheckInBasis_j]
                                    Binv = CheckBinv.copy()
                                    # X=np.dot(CheckBinv,b)
                                    X = np.copy(CheckInMaxX)
                                    Zold = Z
                                    Z = np.dot(CB, X)
                                    DeltaZ = Z - Zold
                                    RC = C - np.dot(CB, np.dot(Binv, A))
                                    if DeltaZ >= MaxDeltaZ:
                                        MaxDeltaZ = DeltaZ
                                        MaxBasis = Basis.copy()
                                        MaxB = A[:, MaxBasis]
                                        MaxCB[0, :] = C[0, MaxBasis]

                                    # K iteration
                                    if Pivots > 2:
                                        for k in range(0, Q):
                                            if QMatrix[i, k] * QMatrix[j, k] == 1:
                                                QMatrix[k, i] = 0
                                                KCutCondition = (
                                                    QMatrix[i, Q] >= MaxDeltaZ + eps
                                                    if Cut
                                                    else True
                                                )
                                                if KCutCondition:
                                                    stats["NbrNodes"] += 1
                                                    stats["NbrNodes_k"] += 1
                                                    # RCk=RC[0,Positive_RC_Indexes[k]]
                                                    TempBasis_k = TempBasis_j.copy()

                                                    test3 = Leaving_Rows_Indexes[k]
                                                    TempBasis_k[test3] = (
                                                        Positive_RC_Indexes[k]
                                                    )

                                                    # Generate Aijk Matrix
                                                    # Generate Bi, Bj and Bk matrix
                                                    # Aijk = A[:, TempBasis_k]
                                                    Bik = A[:, TempBasis_k]
                                                    Bjk = A[:, TempBasis_k]
                                                    Bk = A[:, TempBasis_k]
                                                    if np.linalg.det(Bk) == 0:
                                                        continue
                                                    CB[0, :] = C[0, TempBasis_k]
                                                    for r in range(0, m):
                                                        if (
                                                            TempBasis_k[r]
                                                            == Positive_RC_Indexes[i]
                                                        ):
                                                            for t in range(0, m):
                                                                Bik[t:, r] = b[t]

                                                        if (
                                                            TempBasis_k[r]
                                                            == Positive_RC_Indexes[j]
                                                        ):
                                                            for t in range(0, m):
                                                                Bjk[t:, r] = b[t]

                                                    DeltaB = Bk - TempB
                                                    Dim = LA.matrix_rank(DeltaB)
                                                    stats[f"Rank_{Dim}"] = (
                                                        stats[f"Rank_{Dim}"] + 1
                                                    )
                                                    U, V = CR_factorization(DeltaB)

                                                    # CheckBinv:int=np.zeros((m,m))

                                                    # CheckBinv=Binv+np.dot(np.dot(np.dot(Binv,U),inv(np.eye(Dim)-np.dot(np.dot(V,Binv),U))),np.dot(V,Binv))
                                                    CheckBinv = (
                                                        sherman_morrison_woodbury(
                                                            Binv, Dim, U, V, Bk
                                                        )
                                                    )
                                                    # print("CheckBin: ",CheckBinv)
                                                    # MaxX=np.dot(TempBinv,b)
                                                    CheckInMaxX = np.dot(CheckBinv, b)
                                                    stats["NbrIversions"] += (
                                                        Dim * Dim
                                                    ) / (m * m)
                                                    Feasible = True
                                                    Feasible = np.all(
                                                        CheckInMaxX[:, 0] >= 0
                                                    )
                                                    if not Feasible:
                                                        Basis = TempBasis_k.copy()
                                                        B = A[:, TempBasis_k]
                                                        CB[0, :] = C[0, TempBasis_k]

                                                        Binv = (
                                                            CheckBinv.copy()
                                                        )  # inv(B)
                                                        X = np.dot(Binv, b)
                                                        Zold = Z
                                                        Z = np.dot(CB, X)
                                                        DeltaZ = Z - Zold
                                                        RC = C - np.dot(
                                                            CB, np.dot(Binv, A)
                                                        )

                                                        if DeltaZ >= MaxDeltaZ:
                                                            MaxDeltaZ = DeltaZ
                                                            MaxBasis = Basis.copy()
                                                            MaxB = A[:, MaxBasis]
                                                            MaxCB[0, :] = C[0, MaxBasis]

                                                    else:
                                                        stats["NbrInfeasibilities"] += 1

                                else:
                                    stats["NbrInfeasibilities"] += 1  # here

                            else:
                                stats["NbrCuts"] += 1
            else:
                stats["NbrCuts"] += 1

        Basis = MaxBasis.copy()
        B = A[:, Basis]
        CB[0, :] = C[0, Basis]
        Binv = inv(B)
        stats["NbrIversions"] += 1
        X = np.dot(Binv, b)
        Z = np.dot(CB, X)
        RC = C - np.dot(CB, np.dot(Binv, A))
        MaxRC = np.max([0] + [rc for rc in RC[0, :] if rc > eps])
        isFeasible = np.all(X >= 0)

    X = np.dot(inv(B), b)
    Z = np.dot(CB, X)
    stats["Message"] = "Solved"
    stats["Time"] = time.perf_counter() - tic
    return Z[0][0], X, RC, stats


if __name__ == "__main__":
    print("Welcome to multipivot_simplex_solver.py")
