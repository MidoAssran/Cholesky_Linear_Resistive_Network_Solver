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
from utils import matrix_transpose

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


        # ----------------------------------------------------------------------------------------------------------- #
        # ----------- Simultaneous cholesky factorization of A and solving of the lower traingular system ----------- #
        # ----------------------------------------------------------------------------------------------------------- #
        # Cholesky factorization & forward substitution
        for j in range(n):

            # If the matrix A is not positive definite, exit
            if A[j,j] <= 0:
                return None

            A[j,j] = A[j,j] ** 0.5      # Compute the j,j entry of chol(A) and overwrite A
            b[j] /= A[j,j]              # Compute the j entry of the solution vector being solved for and overwrite b


            for i in range(j+1, n):
                A[i,j] /= A[j,j]        # Compute the i,j entry of chol(A) and overwritte A
                b[i] -= A[i,j] * b[j]   # Look ahead modification of b

                # Look ahead moidification of A
                for k in range(j+1, i+1):
                    A[i,k] -= A[i,j] * A[k,j]
        # ----------------------------------------------------------------------------------------------------------- #


        # ----------------------------------------------------------------------------------------------------------- #
        # ---------------------------------- Now solve the upper traingular system ---------------------------------- #
        # ----------------------------------------------------------------------------------------------------------- #
        # Transpose(A) is the upper-tiangular matrix of the overwritten cholesky factorization
        A[:] = matrix_transpose(A)

        # Backward substitution
        for j in range(n - 1, -1, -1):
            b[j] /= A[j,j]

            for i in range(j):
                b[i] -= A[i,j] * b[j]
        # ----------------------------------------------------------------------------------------------------------- #

        # The solution was overwritten in the vector b
        return b

if __name__ == "__main__":
    from utils import generate_positive_semidef, matrix_dot_vector

    order = 10
    seed = 5

    print("\n", end="\n")
    print("# ------------------------------------- TEST -------------------------------------- #", end="\n")
    print("# ----------------------------- Cholesky Decomposition ---------------------------- #", end="\n")
    print("# --------------------------------------------------------------------------------- #", end ="\n\n")
    # Create a symmetric, real, positive definite matrix.
    A = generate_positive_semidef(order=order, seed=seed)
    x = np.random.randn(order)
    b = matrix_dot_vector(A=A, b=x)
    print("A:\n", A, end="\n\n")
    print("x:\n", x, end="\n\n")
    print("b (=Ax):\n", b, end="\n\n")
    chol_d = CholeskyDecomposition()
    v = chol_d.solve(A=A, b=b)
    print("chol_d.solve(A, b):\n", v, end="\n\n")
    print("2-norm error:\n", np.linalg.norm(v - x), end="\n\n")
    print("# --------------------------------------------------------------------------------- #", end ="\n\n")
