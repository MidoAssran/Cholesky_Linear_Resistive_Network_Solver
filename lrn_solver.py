# ----------------------------------------- #
# Linear Resistive Network Solver
# ----------------------------------------- #
# Author: Mido Assran
# Date: 30, September, 2016
# Description: LinearResistiveNetworkSolver reads a CSV description of
# a linear resistive network, and determines all the node voltages
# of the circuit by constructing a linear system of equations,
# and solving the system using Choleski Decomposition.

import random
import csv
import numpy as np
from choleski import CholeskiDecomposition
from utils import matrix_transpose, matrix_dot_matrix, matrix_dot_vector, vector_to_diag

DEBUG = False

class LinearResistiveNetworkSolver(object):

    #-----Instance Variables-------#
    # _A -> The matrix 'A' in the system of equations Ax = b
    # _b -> The vector 'b' in the system of equations Ax = _b

    def __init__(self, fname):
        """
        :type fname: String
        :rtype: void
        """
        if DEBUG:
            np.core.arrayprint._line_width = 200

        #-----Load data from file-----#
        # Program first reads branch data, then swtiches to reading the
        # incidence matrix. Flag goes high when the the program
        # swtiches to reading the incidence matrix.
        flag = False
        network_branches = []
        incidence_matrix = []
        reader = csv.reader(open(fname, 'r'))
        for row in reader:
            if len(row) == 1 and row[0] == ".":
                flag = True
                continue
            elif len(row) == 0:
                continue
            if not flag:
                network_branches += [list(row)]
            else:
                incidence_matrix += [list(row)]
        network_branches = np.array(network_branches, dtype=np.float64)
        incidence_matrix = np.array(incidence_matrix, dtype=np.float64)
        J = network_branches[:, 0]
        Y = vector_to_diag(1 / network_branches[:, 1])
        E = network_branches[:, 2]
        A = matrix_dot_matrix(A=matrix_dot_matrix(A=incidence_matrix, B=Y),
                              B=matrix_transpose(incidence_matrix))
        b = matrix_dot_vector(A=incidence_matrix,
                              b=(J - matrix_dot_vector(A=Y, b=E)))
        self._A = A
        self._b = b

    def solve(self):
        """
        :rtype: numpy.array([float64])
        """
        chol_decomp = CholeskiDecomposition()
        # Choleski decomposition will overwrite A, and b
        return chol_decomp.solve(A=self._A, b=self._b)


    @staticmethod
    def create_lrn_mesh_data(N, fname):
        """
        :type N: int
        :type fname: String
        :rtype: void
        """
        num_nodes = (N + 1) ** 2
        num_branches = 2 * (N ** 2) + 2 * N + 1
        incidence_matrix = np.empty([num_nodes, num_branches])
        network_branches = np.empty([num_branches, 3])
        incidence_matrix[:] = 0
        network_branches[:] = 0

        for i, row in enumerate(network_branches):
            if i == (num_branches - 1):
                network_branches[i, :] = np.array([0, 1, 1])
            else:
                network_branches[i, :] = np.array([0, 1e3, 0])

        node_num = 0

        # Iterate through node rows of mesh
        for level in range(N + 1):

            # Iterate through node columns of mesh
            for column in range(N + 1):

                # If the node has a left branch
                if (node_num % (N + 1) != 0):
                    left_branch = node_num + (level * N) - 1
                    incidence_matrix[node_num, left_branch] = -1
                    if DEBUG:
                        print("L:", node_num, left_branch, end="\t")

                # If the node has a right branch
                if ((node_num + 1) % (N + 1) != 0):
                    right_branch = node_num + (level * N)
                    incidence_matrix[node_num, right_branch] = 1
                    if DEBUG:
                        print("R:", node_num, right_branch, end="\t")

                # If the node has a top branch
                if (node_num < (num_nodes - (N + 1))):
                    top_branch = node_num + ((level + 1) * N)
                    incidence_matrix[node_num, top_branch] = 1
                    if DEBUG:
                        print("T:", node_num, top_branch, end="\t")

                # If the node has a botom branch
                if (node_num > N):
                    bottom_branch = (node_num - 1) + ((level - 1) * N)
                    incidence_matrix[node_num, bottom_branch] = -1
                    if DEBUG:
                        print("B:", node_num, bottom_branch, end="\t")

                if DEBUG:
                    print("\n")

                node_num += 1

        # Add the branch of the test source
        incidence_matrix[0, -1] = -1
        incidence_matrix[-1, -1] = 1

        # Write data to file fname.csv
        fwriter = csv.writer(open(fname, 'w'))
        for i, row in enumerate(network_branches):
            fwriter.writerow(row)

        # Write a period to separate network_branches from
        # the incidence_matrix
        fwriter.writerow(".")

        for i, row in enumerate(incidence_matrix):
            fwriter.writerow(row)




if __name__ == "__main__":
    print("\n", end="\n")
    print("# -------------------- TEST -------------------- #", end="\n")
    print("# ------- Linear Resistive Network Solver ------ #", end="\n")
    print("# --------------- Manual CSV Data -------------- #", end="\n")
    print("# ---------------------------------------------- #", end="\n\n")
    lrn = LinearResistiveNetworkSolver("data/test_c1.csv")
    voltages = lrn.solve()
    print("Voltages:", end="\n")
    for i, v in enumerate(voltages):
        print(" Node", i, end=": ")
        print(v, "Volts", end="\n")
    print("\n", end="\n")

    print("# -------------------- TEST -------------------- #", end="\n")
    print("# ------- Linear Resistive Network Solver ------ #", end="\n")
    print("# ------------ Finite Difference Mesh ---------- #", end="\n")
    print("# ---------------------------------------------- #", end="\n\n")
    new_fname = "data/test_save.csv"
    N = 15
    print("Mesh size:\n", N, "x", N, end="\n\n")
    LinearResistiveNetworkSolver.create_lrn_mesh_data(N=N, fname=new_fname)
    lrn = LinearResistiveNetworkSolver(new_fname)
    voltages = lrn.solve()
    r_eq = (voltages[0] - voltages[-1])/ (1 - (voltages[0] - voltages[-1]))
    print("Resistance:\n", r_eq, "Ohms", end="\n\n")
