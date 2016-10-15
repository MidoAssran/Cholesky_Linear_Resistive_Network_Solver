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
from conductor_description import *

DEBUG = True

class FiniteDifferencePotentialSolver(object):

    """
    :-------Instance Variables-------:
    :type _h: float -> The inter-mesh node spacing
    :type _num_x_points: float -> The number of mesh points in the x direction
    :type _num_y_points: float -> The number of mesh points in the y direction
    :type _potentials: np.array([float]) -> The electric potential at every point in the mesh
    """

    def __init__(self, h):
        """
        :type h: float
        :rtype: void
        """

        if DEBUG:
            np.core.arrayprint._line_width = 200

        self._h = h

        x_midpoint = INNER_COORDINATES[0] + INNER_HALF_DIMENSIONS[0]
        y_midpoint = INNER_COORDINATES[1] + INNER_HALF_DIMENSIONS[1]
        self._num_x_points = int(x_midpoint / h + 1)
        self._num_y_points = int(y_midpoint / h + 1)

        self._right_spacing_matrix = np.empty((self._num_x_points - 1, self._num_y_points))
        self._left_spacing_matrix = np.empty((self._num_x_points - 1, self._num_y_points))
        self._bottom_spacing_matrix = np.empty((self._num_x_points, self._num_y_points - 1))
        self._top_spacing_matrix = np.empty((self._num_x_points, self._num_y_points - 1))

        # Create equal node spacings
        self._right_spacing_matrix[:] = 1
        self._left_spacing_matrix[:] = 1
        self._top_spacing_matrix[:] = 1
        self._bottom_spacing_matrix[:] = 1

        # # Create unequal row spacings
        # normalizer_x = self._num_x_points - 0; normalizer_y = self._num_y_points - 3.5
        # self.create_unequal_node_spacing_matrix_row(self._right_spacing_matrix, x_midpoint, normalizer_x)
        # self._left_spacing_matrix[:] = self._right_spacing_matrix[:]
        # # Create unequal column spacings
        # self.create_unequal_node_spacing_matrix_column(self._bottom_spacing_matrix, y_midpoint, normalizer_y)
        # self._top_spacing_matrix[:] = self._bottom_spacing_matrix[:]

        # Create boundaries
        z = np.empty((1, self._num_y_points))
        z[:] = self._right_spacing_matrix[-1, 0]
        self._right_spacing_matrix = np.append(self._right_spacing_matrix, z, axis=0)
        self._left_spacing_matrix = np.append(z, self._left_spacing_matrix, axis=0)
        z = np.empty((self._num_x_points, 1))
        z[:] = self._top_spacing_matrix[0, -1]
        self._top_spacing_matrix = np.append(self._top_spacing_matrix, z, axis=1)
        self._bottom_spacing_matrix = np.append(z, self._bottom_spacing_matrix, axis=1)

        # Initialize potentials matrix according to the boundary coniditions
        potentials = np.empty((self._num_x_points, self._num_y_points))
        potentials[:] = 0
        for i in range(self._num_x_points):
            for j in range(self._num_y_points):
                coordinates = self.map_indices_to_coordinates((i,j))
                # If in conductor set potential to conductor potential
                if (coordinates[0] >= INNER_COORDINATES[0]) and (coordinates[1] >= INNER_COORDINATES[1]):
                    potentials[i, j] = CONDUCTOR_POTENTIAL
        self._potentials = potentials

        if DEBUG:
            print(self._right_spacing_matrix)
            # for i in range(self._num_x_points):
            #     for j in range(self._num_y_points):
            #         print(self.map_indices_to_coordinates((i,j)))


    def create_unequal_node_spacing_matrix_column(self, fill_in_matrix, edge_length, normalizer):
        """
        :type fill_in_matrix: np.array([float])
        :rtype: void
        """
        for i in range(fill_in_matrix.shape[1]):
            column = fill_in_matrix[:, i]
            normalizer = normalizer
            sum_sub_column = (((len(column) * (len(column) + 1)) / 2) - len(column)) / normalizer

            column[:] = np.array([i / normalizer for i in range(len(column), 0, -1)])

            # Rebalance the first element in the row to make sure node spacing still spans the physical size of the structure
            column[0] = (edge_length - sum_sub_column * self._h) / self._h


    def create_unequal_node_spacing_matrix_row(self, fill_in_matrix, edge_length, normalizer):
        """
        :type fill_in_matrix: np.array([float])
        :rtype: void
        """
        for i, row in enumerate(fill_in_matrix):
            normalizer = normalizer
            sum_sub_row = (((len(row) * (len(row) + 1)) / 2) - len(row)) / normalizer

            # Create smaller mesh spacing towards the end of the row, and larger towards the beginning
            row[:] = np.array([i / normalizer for i in range(len(row), 0, -1)])

            # Rebalance the first element in the row to make sure node spacing still spans the physical size of the structure
            row[0] = (edge_length - sum_sub_row * self._h) / self._h

    # Helper function that converts node indices to physical locations in the mesh
    def map_indices_to_coordinates(self, indices):
        """
        :type indices: (int, int)
        :rtype: (float, float)
        """
        x, y = 0, 0
        for i in range(indices[0]):
            x += self._right_spacing_matrix[0, i]
        x *= self._h

        for i in range(indices[1]):
            y += self._top_spacing_matrix[i, 0]
        y *= self._h

        coordinates = (x , y)
        return coordinates


    # Helper function that converts node locations in the mesh to indices
    def map_coordinates_to_indices(self, coordinates):
        """
        :type coordinates: (float, float)
        :rtype: (int, int)
        """
        i, j = 0, 0
        x, y = 0, 0

        while x < coordinates[0]:
            x += self._right_spacing_matrix[i, 0] * self._h
            i += 1

        while y < coordinates[1]:
            y += self._top_spacing_matrix[0, j] * self._h
            j += 1

        indices = (i, j)
        return indices


    # Solve for potentials using Successive Over Relaxation
    def solve_sor(self, max_residual, omega=1.5):
        """
        :type max_residual: float
        :type omega: float
        :rtype: (int, np.array([float]))
        """

        if DEBUG:
            print("# --------------------------------------------------------------------------------- #", end ="\n")
            print("# Solving using Successive Over Relaxation Method:", end="\n")
            print("# --------------------------------------------------------------------------------- #", end ="\n\n")

            if omega == 1:
                print("# --------------------------------------------------------------------------------- #", end ="\n")
                print("# Warning, the provided relaxation parameter reduces the method to Gauss-Seidl", end="\n")
                print("# --------------------------------------------------------------------------------- #", end ="\n\n")

        residual = np.empty((self._num_x_points, self._num_y_points))
        condition = True
        itr = 0

        while condition:

            itr += 1

            # Update the potentials
            for i in range(self._num_x_points):
                for j in range(self._num_y_points):

                    # If at a defined point (held at a fixed potential), skip updating this node
                    coordinates = self.map_indices_to_coordinates((i,j))
                    if (i == 0) or (j == 0) or ((coordinates[0] >= INNER_COORDINATES[0]) and (coordinates[1] >= INNER_COORDINATES[1])):
                        continue

                    # Determine adjacent node values: if at boundary apply boundary conditions, else just get adjacent node values
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

                    # Determine the constants induced by unequal node spacings(will cancel out if spacings are equal)
                    c_top, c_bottom, c_left, c_right, c_center = 0, 0, 0, 0, 0
                    sp_t = self._top_spacing_matrix[i, j]
                    sp_b = self._bottom_spacing_matrix[i, j]
                    sp_l = self._left_spacing_matrix[i, j]
                    sp_r = self._right_spacing_matrix[i, j]
                    c_center = 1 + (sp_l / sp_r) + ((sp_l * (sp_l + sp_r)) / (sp_t * (sp_t + sp_b))) + ((sp_l * (sp_l + sp_r)) / (sp_b * (sp_t + sp_b)))
                    c_left = 1
                    c_right = (sp_l / sp_r)
                    c_bottom =  ((sp_l * (sp_l + sp_r)) / (sp_t * (sp_t + sp_b)))
                    c_top = ((sp_l * (sp_l + sp_r)) / (sp_b * (sp_t + sp_b)))

                    # Perform update of potential
                    gauss_seidl = (1.0 / c_center) * (c_top * top + c_bottom * bottom + c_left * left + c_right * right)
                    self._potentials[i, j] =  (1 - omega) * self._potentials[i, j] + omega * gauss_seidl


            # Update the residual
            for i in range(self._num_x_points):
                for j in range(self._num_y_points):

                    # If at a defined point (held at a fixed potential), skip computing this residual and fix at zero
                    coordinates = self.map_indices_to_coordinates((i,j))
                    if (i == 0) or (j == 0) or ((coordinates[0] >= INNER_COORDINATES[0]) and (coordinates[1] >= INNER_COORDINATES[1])):
                        residual[i, j] = 0
                        continue

                    # Determine adjacent node values: if at boundary apply boundary conditions, else just get adjacent node values
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

                    # Determine the constants induced by unequal node spacings(will cancel out if spacings are equal)
                    c_top, c_bottom, c_left, c_right, c_center = 0, 0, 0, 0, 0
                    sp_t = self._top_spacing_matrix[i, j]
                    sp_b = self._bottom_spacing_matrix[i, j]
                    sp_l = self._left_spacing_matrix[i, j]
                    sp_r = self._right_spacing_matrix[i, j]
                    c_center = 1 + (sp_l / sp_r) + ((sp_l * (sp_l + sp_r)) / (sp_t * (sp_t + sp_b))) + ((sp_l * (sp_l + sp_r)) / (sp_b * (sp_t + sp_b)))
                    c_left = 1
                    c_right = (sp_l / sp_r)
                    c_bottom =  ((sp_l * (sp_l + sp_r)) / (sp_t * (sp_t + sp_b)))
                    c_top = ((sp_l * (sp_l + sp_r)) / (sp_b * (sp_t + sp_b)))

                    # Perform update of residual
                    residual[i, j] = c_top * top + c_bottom * bottom + c_left * left + c_right * right - c_center * self._potentials[i, j]


            if DEBUG:
                print(self._potentials.astype(int), end="\n\n")

            # Whether or not the residual has become small enough to stop the process
            condition = not(np.all(residual <= max_residual))

        return (itr, self._potentials)


    # Solve for potentials using Jacobi method
    def solve_jacobi(self, max_residual):
        """
        :type max_residual: float
        :rtype: (int, np.array([float]))
        """

        if DEBUG:
            print("# --------------------------------------------------------------------------------- #", end ="\n")
            print("# Solving using Jacobi Method:", end="\n")
            print("# --------------------------------------------------------------------------------- #", end ="\n\n")

        temp = np.empty((self._num_x_points, self._num_y_points))
        residual = np.empty((self._num_x_points, self._num_y_points))
        condition = True
        itr = 0

        while condition:

            itr += 1

            # Update the potentials
            for i in range(self._num_x_points):
                for j in range(self._num_y_points):

                    # If at a defined point (held at a fixed potential), skip updating this node
                    coordinates = self.map_indices_to_coordinates((i,j))
                    if (i == 0) or (j == 0) or ((coordinates[0] >= INNER_COORDINATES[0]) and (coordinates[1] >= INNER_COORDINATES[1])):
                        temp[i, j] = self._potentials[i, j]
                        continue

                    # Determine adjacent node values: if at boundary apply boundary conditions, else just get adjacent node values
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

                    # Determine the constants induced by unequal node spacings(will cancel out if spacings are equal)
                    c_top, c_bottom, c_left, c_right, c_center = 0, 0, 0, 0, 0
                    sp_t = self._top_spacing_matrix[i, j]
                    sp_b = self._bottom_spacing_matrix[i, j]
                    sp_l = self._left_spacing_matrix[i, j]
                    sp_r = self._right_spacing_matrix[i, j]
                    c_center = 1 + (sp_l / sp_r) + ((sp_l * (sp_l + sp_r)) / (sp_t * (sp_t + sp_b))) + ((sp_l * (sp_l + sp_r)) / (sp_b * (sp_t + sp_b)))
                    c_left = 1
                    c_right = (sp_l / sp_r)
                    c_bottom =  ((sp_l * (sp_l + sp_r)) / (sp_t * (sp_t + sp_b)))
                    c_top = ((sp_l * (sp_l + sp_r)) / (sp_b * (sp_t + sp_b)))

                    # Perform update of potentials
                    temp[i, j] = (1.0 / c_center) * (c_top * top + c_bottom * bottom + c_left * left + c_right * right)

            # Only update global potentials here to ensure that the updates are performed using values at the same iteration
            self._potentials[:] = temp[:]

            # Update the residual
            for i in range(self._num_x_points):
                for j in range(self._num_y_points):

                    # If at a defined point (held at a fixed potential), skip computing this residual and fix at zero
                    coordinates = self.map_indices_to_coordinates((i,j))
                    if (i == 0) or (j == 0) or ((coordinates[0] >= INNER_COORDINATES[0]) and (coordinates[1] >= INNER_COORDINATES[1])):
                        residual[i, j] = 0
                        continue

                    # Determine adjacent node values: if at boundary apply boundary conditions, else just get adjacent node values
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

                    # Determine the constants induced by unequal node spacings(will cancel out if spacings are equal)
                    c_top, c_bottom, c_left, c_right, c_center = 0, 0, 0, 0, 0
                    sp_t = self._top_spacing_matrix[i, j]
                    sp_b = self._bottom_spacing_matrix[i, j]
                    sp_l = self._left_spacing_matrix[i, j]
                    sp_r = self._right_spacing_matrix[i, j]
                    c_center = 1 + (sp_l / sp_r) + ((sp_l * (sp_l + sp_r)) / (sp_t * (sp_t + sp_b))) + ((sp_l * (sp_l + sp_r)) / (sp_b * (sp_t + sp_b)))
                    c_left = 1
                    c_right = (sp_l / sp_r)
                    c_bottom =  ((sp_l * (sp_l + sp_r)) / (sp_t * (sp_t + sp_b)))
                    c_top = ((sp_l * (sp_l + sp_r)) / (sp_b * (sp_t + sp_b)))


                    # Perform update of residual
                    residual[i, j] = c_top * top + c_bottom * bottom + c_left * left + c_right * right - c_center * self._potentials[i, j]


            if DEBUG:
                print(self._potentials.astype(int), end="\n\n")

            # Whether or not the residual has become small enough to stop the process
            condition = not(np.all(residual <= max_residual))

        return (itr, self._potentials)

if __name__ == "__main__":
    fndps = FiniteDifferencePotentialSolver(h=0.01)
    num_itr, potentials = fndps.solve_jacobi(max_residual=1e-10)
    # num_itr, potentials = fndps.solve_sor(max_residual=1e-5, omega=1.5)
    indices = fndps.map_coordinates_to_indices((0.06, 0.04))
    p = potentials[indices]
    print("num_itr:", num_itr)
    print(fndps.map_indices_to_coordinates(indices), p)
    print("potentials:\n", potentials)
