# ----------------------------------------- #
# Finite Difference Potential Solver
# ----------------------------------------- #
# Author: Mido Assran
# Date: Oct. 8, 2016
# Description: FiniteDifferencePotentialSolver determines the electric
# potential at all points in a finite difference mesh of a coax using
# one of two methods (SOR or Jacobi).

import random
import numpy as np
from fd_methods import FD_METHODS
from conductor_description import *

DEBUG = True

class FiniteDifferencePotentialSolver(object):

    #-----Instance Variables-------#
    # _fd_method -> The finite difference method to be used to solve the problem
    # _h -> The inter-mesh node spacing
    # _num_x_points -> The number of mesh points in the x direction
    # _num_y_points -> The number of mesh points in the y direction
    # _potentials -> The electric potential at every point in the mesh


    def __init__(self, h, method=FD_METHODS.SOR):
        """
        :type h: float
        :type method: FD_Method
        :rtype: void
        """

        self._fd_method = method

        self._h = h

        x_midpoint = INNER_COORDINATES[0] + INNER_HALF_DIMENSIONS[0]
        y_midpoint = INNER_COORDINATES[1] + INNER_HALF_DIMENSIONS[1]

        self._num_x_points = int(x_midpoint / h + 1)
        self._num_y_points = int(y_midpoint / h + 1)

        potentials = np.empty((self._num_x_points, self._num_y_points))
        potentials[:] = 0

        for i in range(self._num_x_points):
            for j in range(self._num_y_points):
                coordinates = self.map_indices_to_coordinates((i,j))
                # If in conductor
                if (coordinates[0] >= INNER_COORDINATES[0]) and (coordinates[1] >= INNER_COORDINATES[1]):
                    potentials[i, j] = 1.1e2

        self._potentials = potentials

        if DEBUG:
            np.core.arrayprint._line_width = 200


    # Getter
    def getPotentials(self):
        return self._potentials


    # Helper function that converts node indices to physical locations in the mesh
    def map_indices_to_coordinates(self, indices):
        """
        :type indices: (int, int)
        :rtype: (float, float)
        """
        return (indices[0] * self._h, indices[1] * self._h)


    # Helper function that converts node locations in the mesh to indices
    def map_coordinates_to_indices(self, coordinates):
        """
        :type coordinates: (float, float)
        :rtype: (int, int)
        """
        return (int(coordinates[0] / self._h), int(coordinates[1] / self._h))


    # Solve for potentials using Successive Over Relaxation
    def solve_sor(self, max_residual, omega=1.5):
        """
        :type max_residual: float
        :type omega: float
        :rtype: int
        """

        if DEBUG:
            print("# --------------------------------------------------------------------------------- #", end ="\n")
            print("# Solving using Successive Over Relaxation Method:", end="\n")
            print("# --------------------------------------------------------------------------------- #", end ="\n\n")

        residual = np.empty((self._num_x_points, self._num_y_points))
        condition = True
        itr = 0

        while condition:

            itr += 1

            # Update the potentials
            for i in range(self._num_x_points):
                for j in range(self._num_y_points):

                    coordinates = self.map_indices_to_coordinates((i,j))
                    if (i == 0) or (j == 0) or ((coordinates[0] >= INNER_COORDINATES[0]) and (coordinates[1] >= INNER_COORDINATES[1])):
                        # temp[i, j] = self._potentials[i, j]
                        continue

                    top, bottom, left, right = 0, 0, 0, 0

                    if (j + 1) >= self._num_y_points:
                        top = self._potentials[i, j - 1]
                    else:
                        top = self._potentials[i, j + 1]

                    if (i + 1) >= self._num_x_points:
                        right = self._potentials[i - 1, j]
                    else:
                        right = self._potentials[i + 1, j]

                    if (i - 1) < 0:
                        left = 0
                    else:
                        left = self._potentials[i - 1, j]

                    if (j - 1) < 0:
                        bottom = 0
                    else:
                        bottom = self._potentials[i, j - 1]

                    gauss_seidl = 0.25 * (top + bottom + left + right)
                    self._potentials[i, j] =  (1 - omega) * self._potentials[i, j] + omega * gauss_seidl


            # Update the residual
            for i in range(self._num_x_points):
                for j in range(self._num_y_points):

                    coordinates = self.map_indices_to_coordinates((i,j))
                    if (i == 0) or (j == 0) or ((coordinates[0] >= INNER_COORDINATES[0]) and (coordinates[1] >= INNER_COORDINATES[1])):
                        residual[i, j] = 0
                        continue

                    top, bottom, left, right = 0, 0, 0, 0

                    if (j + 1) >= self._num_y_points:
                        top = self._potentials[i, j - 1]
                    else:
                        top = self._potentials[i, j + 1]

                    if (i + 1) >= self._num_x_points:
                        right = self._potentials[i - 1, j]
                    else:
                        right = self._potentials[i + 1, j]

                    if (i - 1) < 0:
                        left = 0
                    else:
                        left = self._potentials[i - 1, j]

                    if (j - 1) < 0:
                        bottom = 0
                    else:
                        bottom = self._potentials[i, j - 1]

                    residual[i, j] = top + bottom + left + right - 4 * self._potentials[i, j]


            if DEBUG:
                print(self._potentials.astype(int), end="\n\n")

            # Whether or not the residual has become small enough to stop the process
            condition = not(np.all(residual <= max_residual))

        return itr


    # Solve for potentials using Jacobi method
    def solve_jacobi(self, max_residual):
        """
        :type max_residual: float
        :rtype: int
        """

        if DEBUG:
            print("# --------------------------------------------------------------------------------- #", end ="\n")
            print("# Solving using Jacobi Method:", end="\n")
            print("# --------------------------------------------------------------------------------- #", end ="\n\n")

        temp = np.empty((self._num_x_points, self._num_y_points))
        print(temp.shape)
        residual = np.empty((self._num_x_points, self._num_y_points))
        condition = True
        itr = 0

        while condition:

            itr += 1

            # Update the potentials
            for i in range(self._num_x_points):
                for j in range(self._num_y_points):

                    coordinates = self.map_indices_to_coordinates((i,j))
                    if (i == 0) or (j == 0) or ((coordinates[0] >= INNER_COORDINATES[0]) and (coordinates[1] >= INNER_COORDINATES[1])):
                        temp[i, j] = self._potentials[i, j]
                        continue

                    top, bottom, left, right = 0, 0, 0, 0

                    if (j + 1) >= self._num_y_points:
                        top = self._potentials[i, j - 1]
                    else:
                        top = self._potentials[i, j + 1]

                    if (i + 1) >= self._num_x_points:
                        right = self._potentials[i - 1, j]
                    else:
                        right = self._potentials[i + 1, j]

                    if (i - 1) < 0:
                        left = 0
                    else:
                        left = self._potentials[i - 1, j]

                    if (j - 1) < 0:
                        bottom = 0
                    else:
                        bottom = self._potentials[i, j - 1]

                    temp[i, j] = 0.25 * (top + bottom + left + right)

            self._potentials[:] = temp[:]

            # Update the residual
            for i in range(self._num_x_points):
                for j in range(self._num_y_points):

                    coordinates = self.map_indices_to_coordinates((i,j))
                    if (i == 0) or (j == 0) or ((coordinates[0] >= INNER_COORDINATES[0]) and (coordinates[1] >= INNER_COORDINATES[1])):
                        residual[i, j] = 0
                        continue

                    top, bottom, left, right = 0, 0, 0, 0

                    if (j + 1) >= self._num_y_points:
                        top = self._potentials[i, j - 1]
                    else:
                        top = self._potentials[i, j + 1]

                    if (i + 1) >= self._num_x_points:
                        right = self._potentials[i - 1, j]
                    else:
                        right = self._potentials[i + 1, j]

                    if (i - 1) < 0:
                        left = 0
                    else:
                        left = self._potentials[i - 1, j]

                    if (j - 1) < 0:
                        bottom = 0
                    else:
                        bottom = self._potentials[i, j - 1]

                    residual[i, j] = top + bottom + left + right - 4 * self._potentials[i, j]


            if DEBUG:
                print(self._potentials.astype(int), end="\n\n")

            # Whether or not the residual has become small enough to stop the process
            condition = not(np.all(residual <= max_residual))

        return itr

    # Generic solve method creating an abstration for the other solver methods
    def solve(self, max_residual, omega=None):
        """
        :type max_residual: float
        :tpe omega: float
        :rtype: num_itr
        """

        if (self._fd_method.value == FD_METHODS.SOR.value):
            return self.solve_sor(max_residual)
        elif (self._fd_method.value == FD_METHODS.Jacobi.value):
            return self.solve_jacobi(max_residual)
        else:
            print("No valid solution method provided")
            return 0


if __name__ == "__main__":
    fndps = FiniteDifferencePotentialSolver(h=0.005, method=FD_METHODS.SOR)
    num_itr = fndps.solve(max_residual=1e-5, omega=1.5)
    potentials = fndps.getPotentials()
    p = potentials[fndps.map_coordinates_to_indices((0.06, 0.04))]
    print("num_itr:", num_itr)
    print("(0.06, 0.04):", p)
