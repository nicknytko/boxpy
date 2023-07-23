"""BoxMG Interpolation."""
import numpy as np
import scipy.sparse as sp
import boxpy.grid


def _convert_to_csr(P_v, P_j):
    ptr = 0
    max_nnz = 0

    # Determine maximum nonzeros in the output
    for row in P_j:
        max_nnz += len(row)

    rowptr = np.empty(len(P_v) + 1, dtype=np.int64)
    indices = np.empty(max_nnz, dtype=np.int64)
    data = np.empty(max_nnz, dtype=np.float64)

    for row in range(len(P_v)):  # pylint: disable=consider-using-enumerate
        rowptr[row] = ptr

        # Combine duplicate entries
        col_argsort = np.argsort(P_j[row])
        v_current = 0

        row_v = P_v[row]
        row_j = P_j[row]

        for row_i, idx in enumerate(col_argsort):
            idx = col_argsort[row_i]

            col = row_j[idx]
            v_current += row_v[idx]

            if ((row_i == len(col_argsort) - 1) or
                    (row_j[idx] != row_j[col_argsort[row_i + 1]])):
                data[ptr] = v_current
                indices[ptr] = col
                v_current = 0
                ptr += 1

    nnz = ptr
    rowptr[len(P_v)] = nnz

    data.resize(nnz)
    indices.resize(nnz)

    return (data, indices, rowptr)


def interpolate_coarsen_2(grid):
    r"""Interpolate and coarsen.

    Parameters
    ----------
    grid : Grid
        Grid object

    Notes
    -----
    Overview of the coarse grid:
    - c points are subsets of the fine grid that are copied directly.
    - gamma points are embedded horizontally or vertically between two c points.
    - iota points are not grid aligned to a c point and interpolate between
    neighboring c and gamma points (including diagonally)

    c---γ---c
    | \ | / |
    γ---ι---γ
    | / | \ |
    c---γ---c

    """
    if grid.ndim != 2:
        raise NotImplementedError('Coarsening by two only implemented ' +
                                  'for 2D problems.')

    if grid.bc != boxpy.grid.BoundaryCondition.DIRICHLET:
        raise NotImplementedError('Coarsening by two only implemented ' +
                                  'for Dirichlet boundary conditions.')

    # Define the coarse grid
    coarse_shape = tuple((dim + 1) // 2 for dim in grid.shape)
    n_coarse = np.prod(coarse_shape)
    grid_coarse = np.prod(grid.shape)

    # Construct interpolation with a row-wise lists of elements
    # P_v[i] has nonzeros for the i'th row, and P_j[i] has column pointers for the i'th row
    # duplicate entries are allowed, and these are summed together upon conversion to CSR
    P_v = [None] * grid_coarse
    P_j = [None] * grid_coarse

    # A few helpers
    def fine_to_coarse_pos(x, y):
        return (x//2, y//2)

    def coarse_pos_to_idx(xc, yc):
        return yc * coarse_shape[0] + xc

    def fine_pos_to_idx(x, y):
        return y * grid.shape[0] + x

    def coarse_pt_in_bounds(xc, yc):
        return (0 <= xc < coarse_shape[0] and
                0 <= yc < coarse_shape[1])

    def fine_pt_in_bounds(x, y):
        return (0 <= x < grid.shape[0] and
                0 <= y < grid.shape[1])

    def add_entry(i, j, v):
        if P_v[i] is None:
            P_v[i] = []
            P_j[i] = []

        P_v[i].append(v)
        P_j[i].append(j)

    def add_scaled_row(i_to, i_from, alpha):
        P_v[i_to].extend(np.array(P_v[i_from]) * alpha)
        P_j[i_to].extend(P_j[i_from])

    # Coarse-points
    for x in range(0, grid.shape[0], 2):
        for y in range(0, grid.shape[1], 2):
            stencil = grid[x, y]
            row = fine_pos_to_idx(x, y)

            xc, yc = fine_to_coarse_pos(x, y)
            col = coarse_pos_to_idx(xc, yc)

            add_entry(row, col, 1.0)  # interpolate exactly with identity

    # Horizontal gamma-points (embedded on x-lines)
    for x in range(1, grid.shape[0], 2):
        for y in range(0, grid.shape[1], 2):
            stencil = grid[x, y]
            row = fine_pos_to_idx(x, y)

            left = -(stencil[-1, 1] + stencil[-1, 0] + stencil[-1, -1])
            right = -(stencil[1, 1] + stencil[1, 0] + stencil[1, -1])
            center = stencil[0, 0] + stencil[0, -1] + stencil[0, 1]

            if center == 0:
                center = 1e-4

            l_xc, l_yc = fine_to_coarse_pos(x-1, y)
            r_xc, r_yc = fine_to_coarse_pos(x+1, y)

            if coarse_pt_in_bounds(l_xc, l_yc):
                l_col = coarse_pos_to_idx(l_xc, l_yc)
                add_entry(row, l_col, left / center)
            if coarse_pt_in_bounds(r_xc, r_yc):
                r_col = coarse_pos_to_idx(r_xc, r_yc)
                add_entry(row, r_col, right / center)

    # Vertical gamma-points (embedded on y-lines)
    for x in range(0, grid.shape[0], 2):
        for y in range(1, grid.shape[1], 2):
            stencil = grid[x, y]
            row = fine_pos_to_idx(x, y)

            top = -(stencil[-1, 1] + stencil[0, 1] + stencil[1, 1])
            bottom = -(stencil[-1, -1] + stencil[0, -1] + stencil[1, -1])
            center = stencil[0, 0] + stencil[-1, 0] + stencil[1, 0]

            if center == 0:
                center = 1e-4

            t_xc, t_yc = fine_to_coarse_pos(x, y+1)
            b_xc, b_yc = fine_to_coarse_pos(x, y-1)

            if coarse_pt_in_bounds(t_xc, t_yc):
                t_col = coarse_pos_to_idx(t_xc, t_yc)
                add_entry(row, t_col, top / center)
            if coarse_pt_in_bounds(b_xc, b_yc):
                b_col = coarse_pos_to_idx(b_xc, b_yc)
                add_entry(row, b_col, bottom / center)

    def iota_try_set(x, y, x_rel, y_rel, c, stencil, row):
        xc, yc = fine_to_coarse_pos(x + x_rel, y + y_rel)

        if coarse_pt_in_bounds(xc, yc):
            col = coarse_pos_to_idx(xc, yc)
            v = -stencil[x_rel, y_rel]

            if ((x_rel == 0 or y_rel == 0) and           # N/S/E/W point
                    fine_pt_in_bounds(x + x_rel, y+y_rel)):
                # Interpolating from a gamma point:
                # Copy the existing row from the matrix.
                add_scaled_row(row, fine_pos_to_idx(x + x_rel, y + y_rel), v / c)
            else:
                # Otherwise, interpolate from the coarse point.
                add_entry(row, col, v/c)

    # Iota-points
    for x in range(1, grid.shape[0], 2):
        for y in range(1, grid.shape[1], 2):

            stencil = grid[x, y]
            row = fine_pos_to_idx(x, y)

            c = stencil[0, 0]
            if c == 0:
                c = 1e-4

            iota_try_set(x, y, -1, -1, c, stencil, row)  # SW
            iota_try_set(x, y,  1, -1, c, stencil, row)  # SE
            iota_try_set(x, y, -1,  1, c, stencil, row)  # NW
            iota_try_set(x, y,  1,  1, c, stencil, row)  # NE

            iota_try_set(x, y,  0, -1, c, stencil, row)  # S
            iota_try_set(x, y, -1,  0, c, stencil, row)  # W
            iota_try_set(x, y,  1,  0, c, stencil, row)  # E
            iota_try_set(x, y,  0,  1, c, stencil, row)  # N

    P = sp.csr_matrix(_convert_to_csr(P_v, P_j), (grid_coarse, n_coarse))

    return P, coarse_shape
