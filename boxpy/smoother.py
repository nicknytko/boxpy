"""Multigrid smoothers for geometric problems."""
import numpy as np
import numpy.linalg as la
import numba
import scipy.sparse as sp
import scipy.sparse.linalg as spla


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
    level : pyamg.multilevel.MultilevelSolver.Level
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


def setup_line_relax(level, direction='x', iterations=2, cycling_down=True):
    """Create a line relaxation smoother in one direction for 2D problems.

    Parameters
    ----------
    level : pyamg.multilevel.MultilevelSolver.Level
      Current level of the multigrid hierarchy
    direction : string, either x or y
      Direction to perform line smoothing in
    iterations : integer
      Number of total smoothing steps to perform
    cycling_down : boolean
      Flag to determine if this is used as a pre or post relaxation method.
      The line relaxation ordering will be flipped for post-relaxation to
      maintain symmetry for, e.g., preconditioning.
    """
    A = level.A
    grid_x, grid_y = level.grid.shape
    grid_numel = grid_x * grid_y

    if direction == 'x':
        R = sp.eye(grid_numel).tocsc()
        ii = grid_x
        jj = grid_y
    else:
        reordering = np.arange(grid_numel).reshape((grid_x, grid_y)).ravel(order='F')
        R = sp.eye(grid_numel).tocsc()[:, reordering]
        ii = grid_y
        jj = grid_x

    # Set this up as a red-black blocked Gauss-Seidel
    # For sweeping across lines parallel to the x-axis, we don't have to do anything
    # special.  To sweep across lines parallel to the y-axis, we rearrange nodes so
    # that diagonal blocks correspond to vertical lines in the grid.

    def _bsr_sweep(Ab, x, b, start):
        for block_row in range(start, Ab.shape[0] // ii, 2):
            D = None
            off_diag_x = np.zeros(ii)
            for i in range(Ab.indptr[block_row], Ab.indptr[block_row + 1]):
                block_col = Ab.indices[i]

                if block_row == block_col:
                    D = Ab.data[i]
                else:
                    off_diag_x -= Ab.data[i] @ x[block_col * ii:(block_col+1) * ii]
            x[block_row * ii : (block_row + 1) * ii] = la.solve(D, b[block_row * ii : (block_row + 1) * ii] + off_diag_x)

    def solver(A, x, b):
        Ab = (R.T@A@R).tobsr((ii, jj))
        xb = R.T@x
        bb = R.T@b
        for _i in range(iterations):
            if cycling_down:
                _bsr_sweep(Ab, xb, bb, 0)
                _bsr_sweep(Ab, xb, bb, 1)
            else:
                _bsr_sweep(Ab, xb, bb, 1)
                _bsr_sweep(Ab, xb, bb, 0)
        x[:] = R@xb

    return solver

def setup_line_relax_xy(level, iterations=1, cycling_down=True):
    """Create a line relaxation smoother in both directions for 2D problems.

    Will perform x followed by y relaxation on the down-cycle, and y followed
    by x relaxation on the up-cycle.

    Parameters
    ---------
    level : pyamg.multilevel.MultilevelSolver.Level
      Current level of the multigrid hierarchy
    iterations : integer
      Number of total smoothing steps to perform
    cycling_down : boolean
      Flag to determine if this is used as a pre or post relaxation method.
      The line relaxation ordering will be flipped for post-relaxation to
      maintain symmetry for, e.g., preconditioning.
    """

    solver_1 = setup_line_relax(level, direction='x', iterations=iterations, cycling_down=cycling_down)
    solver_2 = setup_line_relax(level, direction='y', iterations=iterations, cycling_down=cycling_down)

    if not cycling_down:
        # If we are cycling up, swap the order to retain symmetry
        solver_1, solver_2 = solver_2, solver_1

    def solver(A, x, b):
        solver_1(A, x, b)
        solver_2(A, x, b)

    return solver
