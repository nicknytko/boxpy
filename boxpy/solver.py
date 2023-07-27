"""BoxMG solver."""
import numpy as np
import scipy.sparse.linalg as spla
import pyamg
import pyamg.multilevel
from pyamg.relaxation.smoothing import _unpack_arg, _setup_call
from boxpy.interpolation import interpolate_coarsen_2
from boxpy.smoother import setup_redblack_gauss_seidel


def _create_multilevel_solver(grid,
                              max_levels,
                              min_size,
                              presmoother,
                              postsmoother,
                              coarse_solve,
                              interpolation,
                              restriction):
    """Create a generic BoxMG solver.

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
        a tuple describing the logical shape of the coarse grid.
    restriction : function
        Function that takes a fine grid and interpolation operator and returns
        a restriction operator.

    Returns
    -------
    ml : pyamg.multilevel.MultilevelSolver
        Multilevel object
    """
    levels = []
    current_grid = grid

    pre_type, pre_args = _unpack_arg(presmoother)
    post_type, post_args = _unpack_arg(postsmoother)

    if callable(pre_type):
        setup_presmoother = pre_type
    else:
        setup_presmoother = _setup_call(pre_type)

    if callable(post_type):
        setup_postsmoother = post_type
    else:
        setup_postsmoother = _setup_call(post_type)

    # Recursively define levels in the hierarchy
    while np.all(np.array(current_grid.shape) >= min_size):
        cur_level = pyamg.multilevel.MultilevelSolver.Level()
        P, coarse_size = interpolation(current_grid)
        R = restriction(current_grid, P)

        cur_level.A = current_grid.operator
        cur_level.P = P
        cur_level.R = R
        cur_level.grid = current_grid

        cur_level.presmoother = setup_presmoother(cur_level, **pre_args)
        cur_level.postsmoother = setup_postsmoother(cur_level, **post_args)

        levels.append(cur_level)

        current_grid = current_grid.coarsen(coarse_size, P, R)
        if max_levels is not None and len(levels) >= max_levels - 1:
            break

    # Last level (no restriction nor interpolation)
    cur_level = pyamg.multilevel.MultilevelSolver.Level()
    cur_level.A = current_grid.operator
    cur_level.grid = current_grid

    cur_level.presmoother = setup_presmoother(cur_level, *pre_args)
    cur_level.postsmoother = setup_postsmoother(cur_level, *post_args)

    levels.append(cur_level)

    # Finally, create the multilever solver itself
    return pyamg.multilevel.MultilevelSolver(levels, coarse_solve)


def boxmg_solver(grid, symmetric=None, coarsen_by=2, **kwargs):
    """Blackbox Multigrid (BoxMG) Solver.

    Will detect if problem is symmetric and set-up solver accordingly.

    Parameters
    ----------
    grid : Grid
        Fine grid object

    coarsen_by : integer
        Number of degrees of freedom to coarsen by in each dimension.
        Only supports two for now.

    Returns
    -------
    ml : pyamg.multilevel.MultilevelSolver
        PyAMG solver object
    """
    A = grid.operator
    eps = 1e-4

    # Default arguments

    if 'coarse_solve' not in kwargs:
        kwargs['coarse_solve'] = 'pinv'

    if 'max_levels' not in kwargs:
        kwargs['max_levels'] = None

    if 'min_size' not in kwargs:
        kwargs['min_size'] = 3

    # Check coarsening

    if coarsen_by not in (2, 3):
        raise RuntimeError(f'Unknown value to coarsen by: {coarsen_by}.')

    # TODO: implement coarsening by 3
    if coarsen_by == 3:
        raise NotImplementedError('Coarsening by 3.')

    # Determine if we are a symmetric or nonsymmetric problem, and form
    # interpolation/restriction based on that.

    if symmetric or spla.norm(A.T - A) ** 2.0 < eps:
        # Symmetric problem
        # Form P from A directly, and R=P^T

        sym_interpolate = interpolate_coarsen_2

        def sym_restrict(G, P):  # pylint: disable=unused-argument
            return P.T

        # Use Gauss-Seidel point-smoothing by default
        default_smoother = (setup_redblack_gauss_seidel, {'iterations': 2})

        if 'presmoother' not in kwargs:
            kwargs['presmoother'] = default_smoother
        if 'postsmoother' not in kwargs:
            kwargs['postsmoother'] = default_smoother

        return _create_multilevel_solver(grid,
                                         **kwargs,
                                         interpolation=sym_interpolate,
                                         restriction=sym_restrict)
    else:
        # Nonsymmetric problem:
        # Form P from 1/2(A + A^T)  and R from A^T

        def nonsym_interpolate(G):
            return interpolate_coarsen_2(G.symmetrize())

        def nonsym_restrict(G, P):  # pylint: disable=unused-argument
            RT, _ = interpolate_coarsen_2(G.T)
            return RT.T

        # Use Kaczmarz point-smoothing by default
        default_smoother = ('gauss_seidel_ne', {'iterations': 2})

        if 'presmoother' not in kwargs:
            kwargs['presmoother'] = default_smoother
        if 'postsmoother' not in kwargs:
            kwargs['postsmoother'] = default_smoother

        return _create_multilevel_solver(grid,
                                         **kwargs,
                                         interpolation=nonsym_interpolate,
                                         restriction=nonsym_restrict)
