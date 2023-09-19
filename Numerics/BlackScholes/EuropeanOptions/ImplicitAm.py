import numpy as np
import scipy.linalg as linalg
from .ImplicitEu import ImplicitEu


class ImplicitAmBer(ImplicitEu):

    def _setup_boundary_conditions_(self):
        super(ImplicitEu, self)._setup_boundary_conditions_()
        self.IV = self.grid[np.arange(1, self.M), -1]  # advanced indexing - a copy not a view

    def _traverse_grid_(self):
        P, L, U = linalg.lu(self.coeffs)
        for j in reversed(self.jValues):
            Ux = linalg.solve(L, self.grid[1:-1, j + 1])
            self.grid[1:-1, j] = linalg.solve(U, Ux)
            exerRegion = self.grid[1:-1, j] < self.IV
            self.grid[1:-1, j][exerRegion] = self.IV[exerRegion]
            self.grid[0, j] = 2 * self.grid[1, j] - self.grid[2, j]
            self.grid[-1, j] = 2 * self.grid[-2, j] - self.grid[-3, j]
        return self.grid


class ImplicitAmBerT(ImplicitEu):
    def _setup_boundary_conditions_(self):
        super(ImplicitEu, self)._setup_boundary_conditions_()
        # self.IV = self.grid[np.arange(1,self.M), -1] # advanced indexing - a copy not a view

    def _traverse_grid_(self):
        P, L, U = linalg.lu(self.coeffs)
        self.grid[:, -1] = self.u0
        self.IV = self.grid[np.arange(1, self.M), -1]
        for j in reversed(self.jValues):
            Ux = linalg.solve(L, self.grid[1:-1, j + 1])
            self.grid[1:-1, j] = linalg.solve(U, Ux)
            exerRegion = self.grid[1:-1, j] < self.IV
            self.grid[1:-1, j][exerRegion] = self.IV[exerRegion]
            self.grid[0, j] = 2 * self.grid[1, j] - self.grid[2, j]
            self.grid[-1, j] = 2 * self.grid[-2, j] - self.grid[-3, j]
        return self.grid


class ImplicitAmBreT(ImplicitAmBer):

    def _traverse_grid_(self):

        self.grido = np.zeros(shape=(self.M + 1, self.initial))
        if self.is_call:
            # convert coeffs to upper triangle matrix
            P, L, U = linalg.lu(self.coeffs)
            self.grid[:, -1] = self.grid__[:, -1]
            self.coeffs = U
            for i in range(self.initial):
                for j in reversed(self.jValues):
                    # transform the R.H.S according to the upper triangle conversion
                    self.grid[1:-1, j + 1] = np.dot(np.matrix(L).I, self.grid[1:-1, j + 1])
                    # solve recursively from the bottom
                    for i in reversed(range(1, self.M)):
                        if i == self.M - 1:
                            self.grid[i, j] = self.grid[i, j + 1] / self.coeffs[i - 1, i - 1]
                        else:
                            self.grid[i, j] = (self.grid[i, j + 1] - self.grid[i + 1, j] * self.coeffs[i - 1, i]) / \
                                                  self.coeffs[i - 1, i - 1]
                        self.grid[i, j] = max(self.grid[i, j], self.IV[i - 1])
                    self.grid[0, j] = 2 * self.grid[1, j] - self.grid[2, j]
                    self.grid[-1, j] = 2 * self.grid[-2, j] - self.grid[-3, j]
                self.grido[:, i] = self.grid[:, 0]
                self.grid = np.zeros(shape=(self.M + 1, self.N + 1))
                self.grid[:, -1] = self.grido[:, i]
        else:
            # convert coeffs to lower triangle matrix
            multiplier = np.zeros(self.M - 2)
            for i in reversed(range(1, self.M - 1)):
                multiplier[i - 1] = self.coeffs[i - 1, i] / self.coeffs[i, i]
                self.coeffs[i - 1, :] -= self.coeffs[i, :] * self.coeffs[i - 1, i] / self.coeffs[i, i]

            for j in reversed(self.jValues):
                # transform the R.H.S according to the lower triangle conversion
                for i in reversed(range(1, self.M - 1)):
                    self.grid[i, j + 1] -= self.grid[i + 1, j + 1] * multiplier[i - 1]
                # solve recursively from the top
                for i in range(1, self.M):
                    if i == 1:
                        self.grid[i, j] = self.grid[i, j + 1] / self.coeffs[i - 1, i - 1]
                    else:
                        self.grid[i, j] = (self.grid[i, j + 1] - self.grid[i - 1, j] * self.coeffs[i - 1, i - 2]) / \
                                              self.coeffs[i - 1, i - 1]
                    self.grid[i, j] = max(self.grid[i, j], self.IV[i - 1])
                self.grid[0, j] = 2 * self.grid[1, j] - self.grid[2, j]
                self.grid[-1, j] = 2 * self.grid[-2, j] - self.grid[-3, j]
        return self.grido


class ImplicitAmBre(ImplicitAmBer):
    def _traverse_grid_(self):
        if self.is_call:
            # convert coeffs to upper triangle matrix
            P, L, U = linalg.lu(self.coeffs)
            self.coeffs = U
            for j in reversed(self.jValues):
                # transform the R.H.S according to the upper triangle conversion
                self.grid[1:-1, j + 1] = np.dot(np.matrix(L).I, self.grid[1:-1, j + 1])
                # solve recursively from the bottom
                for i in reversed(range(1, self.M)):
                    if i == self.M - 1:
                        self.grid[i, j] = self.grid[i, j + 1] / self.coeffs[i - 1, i - 1]
                    else:
                        self.grid[i, j] = (self.grid[i, j + 1] - self.grid[i + 1, j] * self.coeffs[i - 1, i]) / \
                                              self.coeffs[i - 1, i - 1]
                    self.grid[i, j] = max(self.grid[i, j], self.IV[i - 1])
                self.grid[0, j] = 2 * self.grid[1, j] - self.grid[2, j]
                self.grid[-1, j] = 2 * self.grid[-2, j] - self.grid[-3, j]
        else:
            # convert coeffs to lower triangle matrix
            multiplier = np.zeros(self.M - 2)
            for i in reversed(range(1, self.M - 1)):
                multiplier[i - 1] = self.coeffs[i - 1, i] / self.coeffs[i, i]
                self.coeffs[i - 1, :] -= self.coeffs[i, :] * self.coeffs[i - 1, i] / self.coeffs[i, i]

            for j in reversed(self.jValues):
                # transform the R.H.S according to the lower triangle conversion
                for i in reversed(range(1, self.M - 1)):
                    self.grid[i, j + 1] -= self.grid[i + 1, j + 1] * multiplier[i - 1]
                # solve recursively from the top
                for i in range(1, self.M):
                    if i == 1:
                        self.grid[i, j] = self.grid[i, j + 1] / self.coeffs[i - 1, i - 1]
                    else:
                        self.grid[i, j] = (self.grid[i, j + 1] - self.grid[i - 1, j] * self.coeffs[i - 1, i - 2]) / \
                                              self.coeffs[i - 1, i - 1]
                    self.grid[i, j] = max(self.grid[i, j], self.IV[i - 1])
                self.grid[0, j] = 2 * self.grid[1, j] - self.grid[2, j]
                self.grid[-1, j] = 2 * self.grid[-2, j] - self.grid[-3, j]
        return self.grid
