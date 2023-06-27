import numpy as np
import scipy.sparse as sp
import enum
import boxpy.grid


def interpolate_coarsen_2(grid):
    if grid.ndim != 2:
        raise NotImplementedError('Coarsening by two only implemented for 2D problems.')

    if grid.bc != boxpy.grid.BoundaryCondition.DIRICHLET:
        raise NotImplementedError('Coarsening by two only implemented for Dirichlet boundary conditions.')

    # Define the coarse grid
    coarse_shape = tuple((dim + 1) // 2 for dim in grid.shape)
    coarse_N = np.prod(coarse_shape)
    grid_N = np.prod(grid.shape)
    P = sp.lil_matrix((grid_N, coarse_N))

    # A few helpers
    def fine_to_coarse_pos(x, y):
        return ((x-1)//2, (y-1)//2)

    def coarse_pos_to_idx(xc, yc):
        return yc * coarse_shape[0] + xc

    def fine_pos_to_idx(x, y):
        return y * grid.shape[0] + x

    def coarse_pt_in_bounds(xc, yc):
        return (xc >= 0 and xc < coarse_shape[0] and
                yc >= 0 and yc < coarse_shape[1])

    def fine_pt_in_bounds(x, y):
        return (x >= 0 and x < grid.shape[0] and
                y >= 0 and y < grid.shape[1])

    # Horizontal gamma-points (embedded on x-lines)
    for x in range(1, grid.shape[0], 2):
        for y in range(0, grid.shape[1], 2):
            stencil = grid[x, y]
            row = fine_pos_to_idx(x, y)

            l = -(stencil[-1, 1] + stencil[-1, 0] + stencil[-1, -1])
            r = -(stencil[1, 1] + stencil[1, 0] + stencil[1, -1])
            c = stencil[0, 0] + stencil[0, -1] + stencil[0, 1]

            if c == 0:
                c = 1.

            l_xc, l_yc = fine_to_coarse_pos(x-1, y)
            r_xc, r_yc = fine_to_coarse_pos(x+1, y)

            if coarse_pt_in_bounds(l_xc, l_yc):
                l_col = coarse_pos_to_idx(l_xc, l_yc)
                P[row, l_col] = l / c
            if coarse_pt_in_bounds(r_xc, r_yc):
                r_col = coarse_pos_to_idx(r_xc, r_yc)
                P[row, r_col] = r / c

    # Vertical gamma-points (embedded on y-lines)
    for x in range(0, grid.shape[0], 2):
        for y in range(1, grid.shape[1], 2):
            stencil = grid[x, y]
            row = fine_pos_to_idx(x, y)

            t = -(stencil[-1, 1] + stencil[0, 1] + stencil[1, 1])
            b = -(stencil[-1, -1] + stencil[0, -1] + stencil[1, -1])
            c = stencil[0, 0] + stencil[-1, 0] + stencil[1, 0]

            if c == 0:
                c = 1.

            t_xc, t_yc = fine_to_coarse_pos(x, y+1)
            b_xc, b_yc = fine_to_coarse_pos(x, y-1)

            if coarse_pt_in_bounds(t_xc, t_yc):
                t_col = coarse_pos_to_idx(t_xc, t_yc)
                P[row, t_col] = t / c
            if coarse_pt_in_bounds(b_xc, b_yc):
                b_col = coarse_pos_to_idx(b_xc, b_yc)
                P[row, b_col] = b / c

    def iota_try_set(x, y, x_rel, y_rel, c, stencil, row):
        xc, yc = fine_to_coarse_pos(x + x_rel, y + y_rel)

        if coarse_pt_in_bounds(xc, yc):
            col = coarse_pos_to_idx(xc, yc)
            v = -stencil[x_rel, y_rel]

            if ((x_rel == 0 or y_rel == 0) and
                fine_pt_in_bounds(x + x_rel, y+y_rel)): # N/S/E/W point
                # Interpolating from a gamma point.  Copy the row from the matrix.
                P[row] += v/c * P[fine_pos_to_idx(x + x_rel, y + y_rel)]
            else:
                # Otherwise, interpolate from the coarse point.
                P[row, col] = v/c

    # Iota-points
    for x in range(1, grid.shape[0], 2):
        for y in range(1, grid.shape[1], 2):

            stencil = grid[x, y]
            row = fine_pos_to_idx(x, y)

            c = stencil[0, 0]
            if c == 0:
                c = 1.

            iota_try_set(x, y, -1, -1, c, stencil, row) # SW
            iota_try_set(x, y,  0, -1, c, stencil, row) # S
            iota_try_set(x, y,  1, -1, c, stencil, row) # SE
            iota_try_set(x, y, -1,  0, c, stencil, row) # W
            iota_try_set(x, y,  1,  0, c, stencil, row) # E
            iota_try_set(x, y, -1,  1, c, stencil, row) # NW
            iota_try_set(x, y,  0,  1, c, stencil, row) # N
            iota_try_set(x, y,  1,  1, c, stencil, row) # NE

    return P, coarse_shape
