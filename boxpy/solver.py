import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyamg
import pyamg.relaxation.relaxation


class BoxMGLevel:
    def __init__(self, P, R):
        if P is not None:
            self.P = P
        if R is not None:
            self.R = R


def extract_stencil(operator, Nx, Ny, i, j):
    def _try_extract(i1, j1, i2, j2):
        if i < 0 or j < 0 or j >= Nx or j >= Ny:
            return 0.
        else:
            try:
                return operator[i1 * Nx + j1, i2 * Nx + j2]
            except:
                return 0.

    stencil = np.zeros((3, 3))
    stencil[0, 0] = _try_extract(i, j, i - 1, j - 1)
    stencil[0, 1] = _try_extract(i, j, i, j - 1)
    stencil[0, 2] = _try_extract(i, j, i + 1, j - 1)
    stencil[1, 0] = _try_extract(i, j, i - 1, j)
    stencil[1, 1] = _try_extract(i, j, i, j)
    stencil[1, 2] = _try_extract(i, j, i + 1, j)
    stencil[2, 0] = _try_extract(i, j, i - 1, j + 1)
    stencil[2, 1] = _try_extract(i, j, i, j + 1)
    stencil[2, 2] = _try_extract(i, j, i + 1, j + 1)
    return stencil


def set_stencil(operator, Nx, Ny, i, j, stencil):
    def _try_set(i1, j1, i2, j2, v):
        if not (i < 0 or j < 0 or j >= Nx or j >= Ny):
            try:
                operator[i1 * Nx + j1, i2 * Nx + j2] = v
            except:
                pass

    _try_set(i, j, i - 1, j - 1, stencil[0, 0])
    _try_set(i, j, i, j - 1, stencil[0, 1])
    _try_set(i, j, i + 1, j - 1, stencil[0, 2])
    _try_set(i, j, i - 1, j, stencil[1, 0])
    _try_set(i, j, i, j, stencil[1, 1])
    _try_set(i, j, i + 1, j, stencil[1, 2])
    _try_set(i, j, i - 1, j + 1, stencil[2, 0])
    _try_set(i, j, i, j + 1, stencil[2, 1])
    _try_set(i, j, i + 1, j + 1, stencil[2, 2])


def extract_cf_stencil(C, i, j):
    return C[i - 1:i + 2, j - 1: j + 2]


class BoxMGSolver:
    def interpolate(self, C, L, r):
        F = np.logical_not(C)
        F_indices = np.where(F)
        C_indices = np.ones_like(C, dtype=np.int64) * -1
        C_indices[C] = np.arange(C.sum())

        P_shape = (L.shape[0], C.sum())
        P = sp.lil_matrix(P_shape)

        ii, jj = np.meshgrid(np.arange(C.shape[0]), np.arange(C.shape[1]))
        ii = ii.flatten()
        jj = jj.flatten()

        stride = C.shape[1]

        # First, do gamma points
        for i, j in zip(ii, jj):
            idx = i * stride + j
            # If we are at a coarse point, return the identity.
            if C[i, j]:
                P[idx, C_indices[i, j]] = 1.
                continue

            f_i = i
            f_j = j
            f_idx = idx

            # Extract stencil and negate all non-center entries
            stencil = extract_stencil(L, self.Nx, self.Ny, f_i, f_j)
            stencil[1, 1] *= -1
            stencil *= -1

            # Extract the neighboring coarse/fine information to determine how to construct interpolation
            cf_stencil = extract_cf_stencil(C, f_i, f_j)

            # Vertical F point
            if cf_stencil[0, 1] and cf_stencil[2, 1]:
                t = stencil[0, :].sum()
                b = stencil[2, :].sum()
                c = stencil[1, 1] - stencil[1, 2] - stencil[1, 0]
                if c == 0:
                    c = 1.

                P[f_idx, C_indices[f_i-1, f_j]] = t / c
                P[f_idx, C_indices[f_i+1, f_j]] = b / c
            # Horizontal F point
            elif cf_stencil[1, 0] and cf_stencil[1, 2]:
                l = stencil[:, 0].sum()
                r = stencil[:, 2].sum()
                c = stencil[1, 1] - stencil[0, 1] - stencil[2, 1]
                if c == 0:
                    c = 1.

                P[f_idx, C_indices[f_i, f_j-1]] = l / c
                P[f_idx, C_indices[f_i, f_j+1]] = r / c

        # Now, do iota points
        for i, j in zip(ii, jj):
            if C[i, j]:
                continue

            idx = i * stride + j
            f_i = i
            f_j = j
            f_idx = idx

            # Extract stencil and negate all non-center entries
            stencil = extract_stencil(L, self.Nx, self.Ny, f_i, f_j)
            stencil[1, 1] *= -1
            stencil *= -1
            cf_stencil = extract_cf_stencil(C, f_i, f_j)

            # Centered F point
            if cf_stencil[0, 0] and cf_stencil[0, 2] and cf_stencil[2, 0] and cf_stencil[2, 2]:
                c = stencil[1, 1]
                if c == 0:
                    c = 1.

                P[f_idx, C_indices[f_i+1, f_j-1]] = stencil[2, 0] / c # SW
                P[f_idx, C_indices[f_i+1, f_j+1]] = stencil[2, 2] / c # SE
                P[f_idx, C_indices[f_i-1, f_j-1]] = stencil[0, 0] / c # NW
                P[f_idx, C_indices[f_i-1, f_j+1]] = stencil[0, 2] / c # NE

                if not cf_stencil[0, 1]:
                    P[f_idx] += stencil[0, 1]/c * P[f_idx - stride] # N
                if not cf_stencil[2, 1]:
                    P[f_idx] += stencil[2, 1]/c * P[f_idx + stride] # S
                if not cf_stencil[1, 0]:
                    P[f_idx] += stencil[1, 0]/c * P[f_idx - 1] # W
                if not cf_stencil[1, 2]:
                    P[f_idx] += stencil[1, 2]/c * P[f_idx + 1] # E

        return P.tocsr()

    def __init__(self, A, Nx, Ny):
        assert(Nx % 2 == 1)
        assert(Ny % 2 == 1)

        self.Nx = Nx
        self.Ny = Ny

        coarse_nodes = np.ones((Nx, Ny), dtype=bool)
        coarse_nodes[1:-1, 1:-1] = False
        coarse_nodes[2:-1:2, 2:-1:2] = True

        r = np.zeros(A.shape[0])
        P = self.interpolate(coarse_nodes, (A + A.T)/2, r)
        R = self.interpolate(coarse_nodes, A.T, r).T

        self.A = A
        self.A_H = R @ A @ P
        self.R = R
        self.P = P

        self.levels = [
            BoxMGLevel(P, R),
            BoxMGLevel(None, None)
        ]

    def solve(self, x, b):
        '''
        Applies one iteration of the multigrid V-cycle.
        '''

        pyamg.relaxation.relaxation.gauss_seidel_ne(self.A, x, b, iterations=2)
        r = b - self.A @ x
        r_H = self.R @ r
        e_H = np.zeros_like(r_H)
        pyamg.relaxation.relaxation.gauss_seidel_ne(self.A_H, e_H, r_H, iterations=2)
        x = x + self.P @ e_H
        pyamg.relaxation.relaxation.gauss_seidel_ne(self.A, x, b, iterations=2)
        return x

    def solve_full(self, x, b, rtol=1e-5, max_iter=500):
        '''
        Solves the linear system to the given (relative) residual tolerance
        '''

        norm_b = la.norm(b)
        if norm_b == 0:
            inv_norm_b = 1
        else:
            inv_norm_b = 1. / norm_b

        res = la.norm(b - self.A @ x)
        res_hist = [res]

        it = 0

        while res * inv_norm_b > rtol:
            x = self.solve(x, b)

            res = la.norm(b - self.A @ x)
            res_hist.append(res)

            it += 1
            if it >= max_iter:
                break

        return x, res_hist
