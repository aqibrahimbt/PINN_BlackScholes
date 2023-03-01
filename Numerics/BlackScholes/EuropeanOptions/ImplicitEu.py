import numpy as np
import scipy.linalg as linalg
from .ExplicitEu import ExplicitEu


class ImplicitEu(ExplicitEu):
    def _setup_coefficients_(self):
        self.alpha = 0.5 * self.dt * (self.r * self.iValues - self.sigma ** 2 * self.iValues ** 2)
        self.beta = self.dt * (self.r + self.sigma ** 2 * self.iValues ** 2)
        self.gamma = -0.5 * self.dt * (self.r * self.iValues + self.sigma ** 2 * self.iValues ** 2)
        self.coeffs = np.diag(self.alpha[1:], -1) + \
                      np.diag(1 + self.beta) + \
                      np.diag(self.gamma[:-1], 1)

    # def _setup_boundary_conditions_(self):
    #     super(ImplicitEu, self)._setup_boundary_conditions_()

    def _traverse_grid_(self):
        P, L, U = linalg.lu(self.coeffs)
        # self.grid[:, -1] = self.u0
        for j in reversed(self.jValues):
            Ux = linalg.solve(L, self.grid[1:-1, j + 1])
            self.grid[1:-1, j] = linalg.solve(U, Ux)
            self.grid[0, j] = 2 * self.grid[1, j] - self.grid[2, j]
            self.grid[-1, j] = 2 * self.grid[-2, j] - self.grid[-3, j]
