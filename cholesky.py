# ----------------------------------------- #
# Cholesky Decomposition
# ----------------------------------------- #
# Author: Mido Assran
# Date: 30, September, 2016
# Description: CholeskyDecomposition solves the linear system of equations: Ax = b by decomposing A
# using Cholesky factorization and using forward and backward substitution to determine x. Matrix A
# must be symmetric, real, and positive definite.

import random
import numpy as np

class CholeskyDecomposition(object):


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

            # If the matrix A is not positive definite, exit
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

        return b


if __name__ == "__main__":
    order = 4
    seed = 1

    print("\n", end="\n")
    print("# ------------------------------------- TEST -------------------------------------- #", end="\n")
    print("# ----------------------------- Cholesky Decomposition ---------------------------- #", end="\n")
    print("# --------------------------------------------------------------------------------- #", end ="\n\n")
    # Create a symmetric, real, positive definite matrix.
    np.random.seed(seed)
    A = np.random.randn(order, order)
    A = A.dot(np.transpose(A))
    x = np.random.randn(order)
    b = A.dot(x)
    print("A:\n", A, end="\n\n")
    print("b:\n", b, end="\n\n")
    chol_d = CholeskyDecomposition()
    v = chol_d.solve(A=A, b=b)
    np.random.seed(seed)
    A = np.random.randn(order, order)
    A = A.dot(np.transpose(A))
    print("x:\n", x, end="\n\n")
    print("Ax:\n", A.dot(v), end="\n\n")
    print("# --------------------------------------------------------------------------------- #", end ="\n\n")
