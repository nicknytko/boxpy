"""Multigrid smoothers for geometric problems."""
import numpy as np
import numba


@numba.njit
def _gauss_seidel_iter(a_data, a_indices, a_indptr, x, b, row):
    """JIT-compiled Gauss-Seidel iteration."""
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


def setup_redblack_gauss_seidel(level, iterations=2, cycling_down=True):
    """Create a red-black Gauss-Seidel smoother for 2D problems.

    Parameters
    ----------
    level : integer
      Current level of the multigrid hierarchy
    iterations : integer
      Number of total smoothing steps to perform
    cycling_down : boolean
      Flag to determine if this is used as a pre or post relaxation method.
      The red-black ordering will be flipped for post-relaxation to maintain
      symmetry for, e.g., if the solver is used as a preconditioner.
    """
    row_count = np.bincount(level.A.tocsc().indices, minlength=level.A.shape[0])
    max_degree = np.max(row_count)

    grid_dim = level.grid.shape

    if cycling_down:
        lstart = 0
        lend = 2
        lstride = 1
    else:
        lstart = 1
        lend = -1
        lstride = -1

    if max_degree > 5:
        # 9-point stencil
        @numba.njit
        def rb_9pt(a_data, a_indices, a_indptr, x, b, grid_dim):
            # Do a 4-coloring of the grid
            for jbeg in range(lstart, lend, lstride):
                for ibeg in range(lstart, lend, lstride):
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
            for jo in range(lstart, lend, lstride):
                for j in range(grid_dim[1]):
                    ioff = (j + jo) % 2
                    for i in range(ioff, grid_dim[0], 2):
                        idx = j * grid_dim[0] + i
                        _gauss_seidel_iter(a_data, a_indices, a_indptr, x, b, idx)

        def solver(A, x, b):
            for _i in range(iterations):
                rb_5pt(A.data, A.indices, A.indptr, x, b, grid_dim)

        return solver
