# ----------------------------------------- #
# Cholesky Decomposition
# ----------------------------------------- #
# Author: Mido Assran
# Date: 30, September, 2016
# Description: CholeskyDecomposition solves the linear system of equations: Ax = b by decomposing matrix A
# using Cholesky factorization and using forward and backward substitution to determine x. Matrix A must
# be symmetric, real, and positive definite.

import random
import numpy as np

class CholeskyDecomposition(object):


    def solve(self, A, b):
        """
        :type A: np.array([float])
        :type b: np.array([float])
        :rtype: np.array([float])
        """

        # If the matrix, A, is not square, exit
        if A.shape[0] != A.shape[1]:
            return None

        n = A.shape[1]

        # Cholesky factorization & forward substitution
        for j in range(n):

            # If the matrix A is not positive definite, exit
            if A[j,j] <= 0:
                return None

            A[j,j] = A[j,j] ** 0.5      # Compute the j,j entry of chol(A) and overwrite
            b[j] /= A[j,j]              # Compute the j entry of the solution vector being solved for


            for i in range(j+1, n):
                A[i,j] /= A[j,j]        # Compute the i,j entry of chol(A) and overwritte
                b[i] -= A[i,j] * b[j]   # Look ahead modification

                # Look ahead moidification
                for k in range(j+1, i+1):
                    A[i,k] -= A[i,j] * A[k,j]

        A[:] = np.tril(A)
        A[:] = np.transpose(A)

        # Backward substitution
        for j in range(n - 1, -1, -1):
            b[j] /= A[j,j]

            for i in range(j):
                b[i] -= A[i,j] * b[j]

        return b


if __name__ == "__main__":
    order = 10
    seed = 5

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
    print("2-norm error:\n", np.linalg.norm(v - x), end="\n\n")
    print("# --------------------------------------------------------------------------------- #", end ="\n\n")
