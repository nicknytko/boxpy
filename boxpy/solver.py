import numpy as np
import pyamg
import pyamg.multilevel
from pyamg.relaxation.smoothing import change_smoothers
from boxpy.interpolation import interpolate_coarsen_2


def boxmg_symmetric_solver(grid,
                           max_levels=None,
                           min_size=5,
                           presmoother=('jacobi', {'iterations': 2}),
                           postsmoother=('jacobi', {'iterations': 2})):
    levels = []
    current_grid = grid

    # Recursively define levels in the hierarchy
    while np.all(np.array(current_grid.shape) >= min_size):
        cur_level = pyamg.multilevel.MultilevelSolver.Level()
        P, coarse_size = interpolate_coarsen_2(current_grid)
        R = P.T

        cur_level.A = current_grid.op
        cur_level.P = P
        cur_level.R = R
        cur_level.grid = current_grid
        levels.append(cur_level)

        current_grid = current_grid.coarsen(coarse_size, P, R)
        if max_levels is not None and len(levels) >= max_levels - 1:
            break

    # Last level (no restriction nor interpolation)
    cur_level = pyamg.multilevel.MultilevelSolver.Level()
    cur_level.A = current_grid.op
    cur_level.grid = current_grid
    levels.append(cur_level)

    ml = pyamg.multilevel.MultilevelSolver(levels, 'pinv')
    change_smoothers(ml, presmoother, postsmoother)

    return ml
