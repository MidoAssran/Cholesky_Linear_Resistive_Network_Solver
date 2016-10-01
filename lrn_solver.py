# ----------------------------------------- #
# Linear Resistive Network Solver
# ----------------------------------------- #
# Author: Mido Assran
# Date: 30, September, 2016
# Description: LinearResistiveNetworkSolver reads a CSV description of
# a linear resistive network, and determines all the node voltages
# of the circuit by constructing a linear system of equations,
# and solving the system using Cholesky Decomposition.

import random
import csv
import numpy as np
from cholesky import CholeskyDecomposition

class LinearResistiveNetworkSolver(object):

    #-----Instance Variables-------#
    # _A -> The matrix 'A' in the system of equations Ax = b
    # _b -> The vector 'b' in the system of equations Ax = _b
    # _x -> The vector of variables 'x' in the system of equations Ax = b

    def __init__(self, fname):
        """
        :type fname: String
        :rtype: void
        """
        # Load data from file
        flag = False # This flag goes high when the the program swtiches to reading the incidence matrix.
        network_branches = []
        incidence_matrix = []
        reader = csv.reader(open(fname, 'r'))
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
        Y = np.diag(1 / network_branches[:, 1])
        E = network_branches[:, 2]
        A = (incidence_matrix.dot(Y).dot(np.transpose(incidence_matrix)))
        b = (incidence_matrix.dot(J - Y.dot(E)))

        self._x = None
        self._A = A
        self._b = b


    def solve(self):
        """
        :rtype: numpy.array([float64])
        """
        chol_decomp = CholeskyDecomposition()
        self._x = chol_decomp.solve(A=self.A, b=self.b) # Will overwrite A, and b
        return self._x



if __name__ == "__main__":
    lrn = LinearResistiveNetworkSolver("test.csv")
    voltages = lrn.solve()
    print("Voltages:", voltages, end="\n\n")
