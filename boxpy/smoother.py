"""Multigrid smoothers for geometric problems."""
import numpy as np
import numba


@numba.njit(cache=True)
def _gauss_seidel_iter(a_data, a_indices, a_indptr, x, b, row):
    """JIT-compiled Gauss-Seidel iteration."""
    r = 0.0
    diag = 1.0

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
        @numba.njit(cache=True)
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
        @numba.njit(cache=True)
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


@numba.njit(cache=True)
def _tridiag_ldlt_factor_inplace(d, e):
    """In-place LDL^T factorization of symmetric tridiagonal.

    d: (n,) diagonal (updated to D)
    e: (n-1,) subdiagonal between i and i+1 (updated to L subdiag)
    """
    n = d.shape[0]
    for i in range(n - 1):
        li = e[i] / d[i]
        e[i] = li
        d[i + 1] = d[i + 1] - (li * li) * d[i]


@numba.njit(cache=True)
def _tridiag_ldlt_solve_inplace(d, e, rhs):
    """Solve (L D L^T) x = rhs in-place on rhs."""
    n = rhs.shape[0]
    # Forward solve with L
    for i in range(1, n):
        rhs[i] = rhs[i] - e[i - 1] * rhs[i - 1]
    # Diagonal solve with D
    for i in range(n):
        rhs[i] = rhs[i] / d[i]
    # Backward solve with L^T
    for i in range(n - 2, -1, -1):
        rhs[i] = rhs[i] - e[i] * rhs[i + 1]


@numba.njit(cache=True)
def _extract_2d_stencil_from_csr(a_data, a_indices, a_indptr, nx, ny):
    """Extract 2D 5-pt/9-pt neighbor coefficients from a CSR matrix.

    Assumes lexicographic ordering with j varying fastest (NumPy C-order for
    arrays shaped (nx, ny)):
      row = i*ny + j

    Returns arrays shaped (nx, ny) for center and each neighbor:
      aO, aW, aE, aS, aN, aSW, aSE, aNW, aNE
    Missing entries are 0.
    """
    aO = np.zeros((nx, ny), dtype=a_data.dtype)
    aW = np.zeros((nx, ny), dtype=a_data.dtype)
    aE = np.zeros((nx, ny), dtype=a_data.dtype)
    aS = np.zeros((nx, ny), dtype=a_data.dtype)
    aN = np.zeros((nx, ny), dtype=a_data.dtype)
    aSW = np.zeros((nx, ny), dtype=a_data.dtype)
    aSE = np.zeros((nx, ny), dtype=a_data.dtype)
    aNW = np.zeros((nx, ny), dtype=a_data.dtype)
    aNE = np.zeros((nx, ny), dtype=a_data.dtype)

    for i in range(nx):
        for j in range(ny):
            row = i * ny + j

            w = row - ny
            e = row + ny
            s = row - 1
            n_ = row + 1
            sw = row - ny - 1
            se = row + ny - 1
            nw = row - ny + 1
            ne = row + ny + 1

            start = a_indptr[row]
            end = a_indptr[row + 1]
            for k in range(start, end):
                col = a_indices[k]
                val = a_data[k]
                if col == row:
                    aO[i, j] = val
                elif col == w:
                    aW[i, j] = val
                elif col == e:
                    aE[i, j] = val
                elif col == s:
                    aS[i, j] = val
                elif col == n_:
                    aN[i, j] = val
                elif col == sw:
                    aSW[i, j] = val
                elif col == se:
                    aSE[i, j] = val
                elif col == nw:
                    aNW[i, j] = val
                elif col == ne:
                    aNE[i, j] = val

    return aO, aW, aE, aS, aN, aSW, aSE, aNW, aNE


def _setup_line_factors(axis, aO, aW, aS):
    """Build and factor tridiagonal systems for each line.

    axis=0: x-lines (vary i, fixed j) => d_fac.shape=(nx, ny), e_fac.shape=(nx-1, ny)
    axis=1: y-lines (vary j, fixed i) => d_fac.shape=(ny, nx), e_fac.shape=(ny-1, nx)
    """
    nx, ny = aO.shape
    if axis == 0:
        d_fac = np.empty((nx, ny), dtype=aO.dtype)
        if nx > 1:
            e_fac = np.empty((nx - 1, ny), dtype=aO.dtype)
        else:
            e_fac = np.empty((0, ny), dtype=aO.dtype)

        d_fac[:, :] = aO[:, :]
        if nx > 1:
            # Use the *subdiagonal* entries A[i+1, i] (west coefficient of row i+1)
            # to match the symmetric tridiagonal convention used by LDL^T.
            e_fac[:, :] = aW[1:, :]
            for j in range(ny):
                _tridiag_ldlt_factor_inplace(d_fac[:, j], e_fac[:, j])
        return d_fac, e_fac

    # axis == 1
    d_fac = np.empty((ny, nx), dtype=aO.dtype)
    if ny > 1:
        e_fac = np.empty((ny - 1, nx), dtype=aO.dtype)
    else:
        e_fac = np.empty((0, nx), dtype=aO.dtype)

    for i in range(nx):
        for j in range(ny):
            d_fac[j, i] = aO[i, j]
        if ny > 1:
            for j in range(ny - 1):
                # Use the *subdiagonal* entries A[j+1, j] (south coefficient of row j+1)
                e_fac[j, i] = aS[i, j + 1]
            _tridiag_ldlt_factor_inplace(d_fac[:, i], e_fac[:, i])

    return d_fac, e_fac


@numba.njit(cache=True)
def _relax_lines_zebra(
    axis,
    x2d,
    b2d,
    aW,
    aE,
    aS,
    aN,
    aSW,
    aSE,
    aNW,
    aNE,
    d_fac,
    e_fac,
    do_9pt,
    cycling_down,
):
    nx, ny = x2d.shape

    if axis == 0:
        # x-lines: choose odd/even rows first depending on cycling direction
        if cycling_down:
            lstart = 1
            lend = 0
            lstride = -1
        else:
            lstart = 0
            lend = 1
            lstride = 1

        tmp = np.empty((nx,), dtype=x2d.dtype)
        for l in range(lstart, lend + lstride, lstride):
            for j in range(l, ny, 2):
                for i in range(nx):
                    v = b2d[i, j]
                    if do_9pt:
                        if j - 1 >= 0:
                            v -= aS[i, j] * x2d[i, j - 1]
                        if j + 1 < ny:
                            v -= aN[i, j] * x2d[i, j + 1]
                        if j - 1 >= 0 and i - 1 >= 0:
                            v -= aSW[i, j] * x2d[i - 1, j - 1]
                        if j - 1 >= 0 and i + 1 < nx:
                            v -= aSE[i, j] * x2d[i + 1, j - 1]
                        if j + 1 < ny and i - 1 >= 0:
                            v -= aNW[i, j] * x2d[i - 1, j + 1]
                        if j + 1 < ny and i + 1 < nx:
                            v -= aNE[i, j] * x2d[i + 1, j + 1]
                    else:
                        if j - 1 >= 0:
                            v -= aS[i, j] * x2d[i, j - 1]
                        if j + 1 < ny:
                            v -= aN[i, j] * x2d[i, j + 1]
                    tmp[i] = v

                _tridiag_ldlt_solve_inplace(d_fac[:, j], e_fac[:, j], tmp)

                for i in range(nx):
                    x2d[i, j] = tmp[i]
        return

    # axis == 1: y-lines
    if cycling_down:
        lstart = 1
        lend = 0
        lstride = -1
    else:
        lstart = 0
        lend = 1
        lstride = 1

    tmp = np.empty((ny,), dtype=x2d.dtype)
    for l in range(lstart, lend + lstride, lstride):
        for i in range(l, nx, 2):
            for j in range(ny):
                v = b2d[i, j]
                if do_9pt:
                    if i - 1 >= 0:
                        v -= aW[i, j] * x2d[i - 1, j]
                    if i + 1 < nx:
                        v -= aE[i, j] * x2d[i + 1, j]
                    if j - 1 >= 0 and i - 1 >= 0:
                        v -= aSW[i, j] * x2d[i - 1, j - 1]
                    if j - 1 >= 0 and i + 1 < nx:
                        v -= aSE[i, j] * x2d[i + 1, j - 1]
                    if j + 1 < ny and i - 1 >= 0:
                        v -= aNW[i, j] * x2d[i - 1, j + 1]
                    if j + 1 < ny and i + 1 < nx:
                        v -= aNE[i, j] * x2d[i + 1, j + 1]
                else:
                    if i - 1 >= 0:
                        v -= aW[i, j] * x2d[i - 1, j]
                    if i + 1 < nx:
                        v -= aE[i, j] * x2d[i + 1, j]
                tmp[j] = v

            _tridiag_ldlt_solve_inplace(d_fac[:, i], e_fac[:, i], tmp)

            for j in range(ny):
                x2d[i, j] = tmp[j]


def setup_line_relaxation_x(level, iterations=2, cycling_down=True):
    """Create a zebra x-line relaxation smoother (Cedar SymStd lines_x).

    This matches the structure of Cedar's `BMG2_SymStd_SETUP_lines_x` and
    `BMG2_SymStd_relax_lines_x`, but operates on this project's interior-only
    grid (no explicit boundary nodes).
    """
    A = level.A.tocsr()
    nx, ny = level.grid.shape
    # Detect stencil width from CSR row nnz (NOT from column index histogram).
    do_9pt = int(np.max(np.diff(A.indptr))) > 5

    aO, aW, aE, aS, aN, aSW, aSE, aNW, aNE = _extract_2d_stencil_from_csr(
        A.data, A.indices, A.indptr, nx, ny
    )

    # Setup: factorize per y-line (each fixed j) the tridiagonal in x.
    d_fac, e_fac = _setup_line_factors(0, aO, aW, aS)

    def solver(A_mat, x, b):  # pylint: disable=unused-argument
        x2d = x.reshape((nx, ny), order="C")
        b2d = b.reshape((nx, ny), order="C")
        for _ in range(iterations):
            _relax_lines_zebra(
                0,
                x2d,
                b2d,
                aW,
                aE,
                aS,
                aN,
                aSW,
                aSE,
                aNW,
                aNE,
                d_fac,
                e_fac,
                do_9pt,
                cycling_down,
            )

    return solver


def setup_line_relaxation_y(level, iterations=2, cycling_down=True):
    """Create a zebra y-line relaxation smoother (Cedar SymStd lines_y)."""
    A = level.A.tocsr()
    nx, ny = level.grid.shape
    do_9pt = int(np.max(np.diff(A.indptr))) > 5

    aO, aW, aE, aS, aN, aSW, aSE, aNW, aNE = _extract_2d_stencil_from_csr(
        A.data, A.indices, A.indptr, nx, ny
    )

    # Setup: factorize per x-line (each fixed i) the tridiagonal in y.
    d_fac, e_fac = _setup_line_factors(1, aO, aW, aS)

    def solver(A_mat, x, b):  # pylint: disable=unused-argument
        x2d = x.reshape((nx, ny), order="C")
        b2d = b.reshape((nx, ny), order="C")
        for _ in range(iterations):
            _relax_lines_zebra(
                1,
                x2d,
                b2d,
                aW,
                aE,
                aS,
                aN,
                aSW,
                aSE,
                aNW,
                aNE,
                d_fac,
                e_fac,
                do_9pt,
                cycling_down,
            )

    return solver
