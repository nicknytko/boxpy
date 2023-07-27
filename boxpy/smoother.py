import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numba


def setup_redblack_gauss_seidel(level, iterations=2):
    row_count = np.bincount(level.A.tocsc().indices, minlength=level.A.shape[0])
    max_degree = np.max(row_count)

    grid_dim = level.grid.shape

    if max_degree > 5:
        # 9-point stencil
        @numba.njit
        def rb_9pt(A_data, A_indices, A_indptr, x, b, grid_dim):
            for jbeg in range(2):
                for ibeg in range(2):
                    for j in range(jbeg, grid_dim[1], 2):
                        for i in range(ibeg, grid_dim[0], 2):
                            idx = j * grid_dim[0] + i

                            r = 0.
                            diag = 2.0 ** -52
                            for k in range(A_indptr[idx], A_indptr[idx + 1]):
                                col = A_indices[k]
                                data = A_data[k]
                                if col == idx:
                                    diag = data
                                else:
                                    r += data * x[col]

                            x[idx] = (b[idx] - r) / diag

        def solver(A, x, b):
            for i in range(iterations):
                rb_9pt(A.data, A.indices, A.indptr, x, b, grid_dim)

        return solver
    else:
        # 5-point stencil
        @numba.njit
        def rb_5pt(A_data, A_indices, A_indptr, x, b, grid_dim):
            for jo in range(2):
                for j in range(grid_dim[1]):
                    ioff = (j + jo) % 2
                    for i in range(ioff, grid_dim[0], 2):
                        idx = j * grid_dim[0] + i

                        r = 0.
                        diag = 2.0 ** -52
                        for k in range(A_indptr[idx], A_indptr[idx + 1]):
                            col = A_indices[k]
                            data = A_data[k]
                            if col == idx:
                                diag = data
                            else:
                                r += data * x[A_indices[k]]

                        x[idx] = (b[idx] - r) / diag

        def solver(A, x, b):
            for i in range(iterations):
                rb_5pt(A.data, A.indices, A.indptr, x, b, grid_dim)

        return solver
