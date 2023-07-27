"""Multigrid smoothers for geometric problems"""
import numpy as np
import numba


@numba.njit
def _gauss_seidel_iter(a_data, a_indices, a_indptr, x, b, row):
    r = 0.
    diag = 1.

    for k in range(a_indptr[row], a_indptr[row + 1]):
        col = a_indices[k]
        data = a_data[k]
        if col == row:
            diag = data
        else:
            r += data * x[col]

    x[row] = (b[row] - r) / diag


def setup_redblack_gauss_seidel(level, iterations=2):
    """Create a red-black Gauss-Seidel smoother for 2D problems."""

    row_count = np.bincount(level.A.tocsc().indices, minlength=level.A.shape[0])
    max_degree = np.max(row_count)

    grid_dim = level.grid.shape

    if max_degree > 5:
        # 9-point stencil
        @numba.njit
        def rb_9pt(a_data, a_indices, a_indptr, x, b, grid_dim):
            # Do a 4-coloring of the grid
            for jbeg in range(2):
                for ibeg in range(2):
                    for j in range(jbeg, grid_dim[1], 2):
                        for i in range(ibeg, grid_dim[0], 2):
                            idx = j * grid_dim[0] + i
                            _gauss_seidel_iter(a_data, a_indices, a_indptr, x, b, idx)

        def solver(A, x, b):
            for _i in range(iterations):
                rb_9pt(A.data, A.indices, A.indptr, x, b, grid_dim)

        return solver
    else:
        # 5-point stencil
        @numba.njit
        def rb_5pt(a_data, a_indices, a_indptr, x, b, grid_dim):
            # Red-black coloring of the grid
            for jo in range(2):
                for j in range(grid_dim[1]):
                    ioff = (j + jo) % 2
                    for i in range(ioff, grid_dim[0], 2):
                        idx = j * grid_dim[0] + i
                        _gauss_seidel_iter(a_data, a_indices, a_indptr, x, b, idx)

        def solver(A, x, b):
            for _i in range(iterations):
                rb_5pt(A.data, A.indices, A.indptr, x, b, grid_dim)

        return solver
