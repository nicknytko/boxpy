"""BoxMG solver."""
import numpy as np
import scipy.sparse.linalg as spla
import pyamg
import pyamg.multilevel
from pyamg.relaxation.smoothing import change_smoothers
from boxpy.interpolation import interpolate_coarsen_2


def _create_multilevel_solver(grid,
                              max_levels,
                              min_size,
                              presmoother,
                              postsmoother,
                              coarse_solve,
                              interpolation,
                              restriction):
    """Creates a generic BoxMG solver.

    Parameters
    ----------
    grid : Grid
        Fine grid object
    max_levels : int
        Maximum number of levels
    min_size : int
        Minimum size for a level
    presmoother : string, tuple
        Presmoother details
    postsmoother : string, tuple
        Postsmoother details
    coarse_solve : string
        Coarse-level solve details
    interpolation : function
        Function that takes a fine grid and returns an interpolation operator and
        a tuple describing the logical shape of the coarse grid..
    restriction : function
        Function that takes a fine grid and interpolation operator and returns
        a restriction operator

    Returns
    -------
    ml : pyamg.multilevel.MultilevelSolver
        Multilevel object
    """

    levels = []
    current_grid = grid

    # Recursively define levels in the hierarchy
    while np.all(np.array(current_grid.shape) >= min_size):
        cur_level = pyamg.multilevel.MultilevelSolver.Level()
        P, coarse_size = interpolation(current_grid)
        R = restriction(current_grid, P)

        cur_level.A = current_grid.operator
        cur_level.P = P
        cur_level.R = R
        cur_level.grid = current_grid
        levels.append(cur_level)

        current_grid = current_grid.coarsen(coarse_size, P, R)
        if max_levels is not None and len(levels) >= max_levels - 1:
            break

    # Last level (no restriction nor interpolation)
    cur_level = pyamg.multilevel.MultilevelSolver.Level()
    cur_level.A = current_grid.operator
    cur_level.grid = current_grid
    levels.append(cur_level)

    ml = pyamg.multilevel.MultilevelSolver(levels, 'pinv')
    change_smoothers(ml, presmoother, postsmoother)

    return ml


def boxmg_symmetric_solver(grid,
                           max_levels=None,
                           min_size=5,
                           presmoother=('gauss_seidel', {'iterations': 2}),
                           postsmoother=('gauss_seidel', {'iterations': 2})):
    """Blackbox Multigrid (BoxMG) Symmetric Solver.

    Parameters
    ----------
    grid : Grid
        Grid object
    max_levels : int
        Maximum number of levels
    min_size : int
        Minimum size for a level
    presmoother : string, tuple
        Presmoother details
    postsmoother : string, tuple
        Postsmoother details

    Returns
    -------
    ml : pyamg.multilevel.MultilevelSolver
        Multilevel object
    """

    return _create_multilevel_solver(grid, max_levels, min_size,
                                     presmoother, postsmoother, 'pinv',
                                     interpolate_coarsen_2, lambda G, P: P.T)


def boxmg_nonsymmetric_solver(grid,
                              max_levels=None,
                              min_size=5,
                              presmoother=('gauss_seidel_ne', {'iterations': 2}),
                              postsmoother=('gauss_seidel_ne', {'iterations': 2})):
    """Blackbox Multigrid (BoxMG) Nonsymmetric Solver.

    Parameters
    ----------
    grid : Grid
        Grid object
    max_levels : int
        Maximum number of levels
    min_size : int
        Minimum size for a level
    presmoother : string, tuple
        Presmoother details
    postsmoother : string, tuple
        Postsmoother details

    Returns
    -------
    ml : pyamg.multilevel.MultilevelSolver
        Multilevel object
    """

    def nonsym_interpolate(G):
        return interpolate_coarsen_2(G.symmetrize())

    def nonsym_restrict(G, P):
        RT, _ = interpolate_coarsen_2(G.T)
        return RT.T

    return _create_multilevel_solver(grid, max_levels, min_size,
                                     presmoother, postsmoother, 'pinv',
                                     nonsym_interpolate, nonsym_restrict)


def boxmg_solver(grid, *args, **kwargs):
    """Blackbox Multigrid (BoxMG) Solver.

    Will detect if problem is symmetric and set-up solver accordingly.

    Parameters
    ----------
    grid : Grid
        Fine grid object

    Returns
    ml : pyamg.multilevel.MultilevelSolver
        PyAMG solver object
    """

    A = grid.operator
    eps = 1e-4
    if spla.norm(A.T - A) ** 2.0 < eps:
        # Symmetric problem
        return boxmg_symmetric_solver(grid, *args, **kwargs)
    else:
        # Nonsymmetric problem
        return boxmg_nonsymmetric_solver(grid, *args, **kwargs)
