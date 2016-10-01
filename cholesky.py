# ----------------------------------------- #
# Cholesky Decomposition
# ----------------------------------------- #
# Author: Mido Assran
# Date: 30, September, 2016
# Description: CholeskyDecomposition solves the linear system of equations: Ax = b by decomposing A
# using Cholesky factorization and using forward and backward substitution to determine x. A must
# be symmetric, real, and positive definite.

import time
import random
import numpy as np
import csv

class CholeskyDecomposition(object):

    #-----Instance Variables-------#
    # _x -> The vector of variables being solved for.

    def __init__(self):
        """
        :rtype: void
        """
        self._x = None


    def chol(self, A):
        """
        :type A: np.array([float])
        :rtype: void
        """

        m = A.shape[0]
        n = A.shape[1]

        # If this matrix isn't square, then it doesn't satisfy the requirements
        # for compatability of cholesky factorization.
        if m != n:
            return None

        for j in range(n):
            A[j,j] = A[j,j] ** 0.5

            for i in range(j+1, m):
                A[i,j] /= A[j,j]

                for k in range(j+1, i):
                    A[i,k] -= A[i,j] * A[k,j]

        # Present the lower triangular half of the overridden matrix.
        A[:] =  np.tril(A)


    def solve(self, A, b):
        """
        :type A: np.array([float])
        :type b: np.array([float])
        :rtype: np.array([float])
        """

        m = A.shape[0]
        n = A.shape[1]

        # If the matrix, A, is not square, exit
        if m != n:
            return None

        for j in range(n):

            # If the matrix, A, is not positive definite, exit
            if A[j,j] <= 0:
                return None

            A[j,j] = A[j,j] ** 0.5      # Compute the j,j entry of chol(A) and overwrite
            b[j] /= A[j,j]              # Compute the j entry of the solution vector being solved for


            for i in range(j+1, m):
                A[i,j] /= A[j,j]        # Compute the i,j entry of chol(A) and overwritte
                b[i] -= A[i,j] * b[j]   # Look ahead modification

                # Look ahead moidification
                for k in range(j+1, i):
                    A[i,k] -= A[i,j] * A[k,j]

if __name__ == "__main__":

    # Start reading the network_branches, then read the incidence_matrix.
    reader = csv.reader(open("test.csv", 'r'))
    flag = False # This flag goes high when the the program swtiches to reading the incidence matrix.
    network_branches = []
    incidence_matrix = []

    for row in reader:
        if len(row) < 1:
            flag = True
            continue

        if not flag:
            network_branches += [list(row)]
        else:
            incidence_matrix += [list(row)]

    network_branches = np.array(network_branches, dtype=np.float64)
    incidence_matrix = np.array(incidence_matrix, dtype=np.int64)

    J = network_branches[:, 0]
    R = 1 / network_branches[:, 1]
    E = network_branches[:, 2]
    print(J, R, E)

    # order = 4
    # seed = 1
    # print("\n", end="\n")
    # print("# ------------------------------------- TEST -------------------------------------- #", end="\n")
    # print("# ----------------------------- Cholesky Factorization ---------------------------- #", end="\n")
    # print("# --------------------------------------------------------------------------------- #", end ="\n")
    # # Create a symmetric, real, positive definite matrix.
    # np.random.seed(seed)
    # A = np.random.randn(order, order)
    # A = A.dot(np.transpose(A))
    # print("A:\n", A, end="\n\n")
    # chol_d = CholeskyDecomposition()
    # chol_d.chol(A)
    # print("chol(A):\n", A, end="\n\n")
    # print("L.dot(transpose(L)):\n", A.dot(np.transpose(A)), end="\n\n")
    # print("# --------------------------------------------------------------------------------- #", end ="\n\n")
    #
    # print("\n", end="\n")
    # print("# ------------------------------------- TEST -------------------------------------- #", end="\n")
    # print("# ----------------------------- Cholesky Decomposition ---------------------------- #", end="\n")
    # print("# --------------------------------------------------------------------------------- #", end ="\n")
    # # Create a symmetric, real, positive definite matrix.
    # np.random.seed(seed)
    # A = np.random.randn(order, order)
    # A = A.dot(np.transpose(A))
    # x = np.random.randn(order)
    # b = A.dot(x)
    # print("b:\n", b, end="\n\n")
    # chol_d = CholeskyDecomposition()
    # chol_d.solve(A=A, b=b)
    # np.random.seed(seed)
    # A = np.random.randn(order, order)
    # A = A.dot(np.transpose(A))
    # print("x:\n", x, end="\n\n")
    # print("Ax:\n", A.dot(x), end="\n\n")
    # print("# --------------------------------------------------------------------------------- #", end ="\n\n")
