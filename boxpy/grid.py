"""Grid and boundary condition classes."""

import enum
import numpy as np
import scipy.sparse as sp


class BoundaryCondition(enum.Enum):
    """Enumerate boundary conditions."""

    DIRICHLET = 1
    PERIODIC = 2


class Grid:
    """Base structured grid class."""

    # pylint: disable-next=too-many-arguments
    def __init__(self, operator, shape, xlim, ylim, bc):
        """Initialize the grid class."""
        self.operator = operator
        self.shape = shape
        self.ndim = len(self.shape)
        self.xlim = xlim
        self.ylim = ylim
        self.bc = bc

    def coarsen(self, new_shape, P, R=None):
        """Coarsen the operator."""
        if R is None:
            R = P.T
        AH = R @ self.operator @ P
        return Grid(AH, new_shape, self.xlim, self.ylim, self.bc)

    def symmetrize(self):
        """Symmetrizes a non-symmetric operator."""
        A_sym = (self.operator + self.operator.T)/2
        return Grid(A_sym, self.shape, self.xlim, self.ylim, self.bc)

    def transpose(self):
        """Transposes a non-symmetric operator."""
        return Grid(self.operator.T, self.shape, self.xlim, self.ylim, self.bc)

    @property
    def T(self):
        """Transposes a non-symmetric operator."""
        return self.transpose()

    def __call__(self, *indices):
        """Retrieve stencils."""
        return Stencil(self, *(np.array(indices) - 1))

    def __getitem__(self, indices):
        """Retrieve a stencil."""
        return Stencil(self, *indices)

    def __setitem__(self, indices, stencil_val):
        """Set stencils."""
        stencil = Stencil(self, *indices)
        stencil[:] = stencil_val

    def interp_fcn(self, f):
        """Interpolate a function on the grid.

        Parameters
        ----------
        f : function
            Function of two parameters, x and y

        Return
        ------
        Grid function
            Evaluation of the function on the grid
        """
        xx, yy = np.meshgrid(
            np.linspace(self.xlim[0], self.xlim[1], self.shape[0]),
            np.linspace(self.ylim[0], self.ylim[1], self.shape[1]))

        return f(xx, yy)


def _eval_bc(A, nx, ny, bc):
    bc_mask = np.zeros((nx+2, ny+2), dtype=bool)

    bc_mask[0] = True
    bc_mask[-1] = True
    bc_mask[:, 0] = True
    bc_mask[:, -1] = True

    int_mask = ~bc_mask

    R = sp.eye((nx + 2) * (ny + 2)).tocsr()
    R = R[int_mask.flatten()]

    if bc is not None:
        x = np.linspace(0, 1, nx + 2)
        y = np.linspace(0, 1, ny + 2)

        xx, yy = np.meshgrid(x, y)

        bc_eval = np.zeros((nx+2, ny+2))
        bc_eval[bc_mask] = bc(xx[bc_mask], yy[bc_mask])

        I_i = sp.diags(int_mask.flatten().astype(np.int64))
        B_i = sp.diags(bc_mask.flatten().astype(np.int64))

        A_bc = I_i@A + B_i

        return R@A@R.T, -R@(A_bc@bc_eval.flatten())
    else:
        return R@A@R.T, np.zeros((nx, ny))


def create_poisson_dirichlet_2d(nx, ny, nu, bc=None):
    """Construct a Poisson operator in 2D.

    Dirichlet boundary conditions are assumed, and boundary nodes are removed
    from the returned operator.

    Parameters
    ----------
    nx : int
        Number of (interior) grid points in x
    ny : int
        Number of (interior) grid points in y
    nu : float
        Scalar diffusion coefficient
    bc : function
        Function to evaluate at the boundary conditions,
        if None is given then homogeneous bc are assumed.

    Returns
    -------
    Grid object
    """
    dx = 1./(nx + 1)
    dy = 1./(ny + 1)

    Ax = (nu * (1./dx**2) * (sp.eye(nx+2) * 2 - sp.eye(nx+2, k=-1) - sp.eye(nx+2, k=1)))
    Ay = (nu * (1./dy**2) * (sp.eye(ny+2) * 2 - sp.eye(ny+2, k=-1) - sp.eye(ny+2, k=1)))

    A = sp.kron(Ax, sp.eye(ny + 2)) + sp.kron(sp.eye(nx + 2), Ay)

    x = np.linspace(0, 1, nx + 2)
    y = np.linspace(0, 1, ny + 2)

    A, b = _eval_bc(A, nx, ny, bc)

    return Grid(A,
                shape=(nx, ny),
                xlim=(x[1], x[-1]),
                ylim=(y[1], y[-1]),
                bc=BoundaryCondition.DIRICHLET), b


def create_advection_dirichlet_2d(nx, ny, nu, v, bc=None):
    """Construct a convection-diffusion problem in 2d discretized with FDM.

    Solves the problem:
    div(nu * grad u) - div(vu) = 0

    Dirichlet boundary conditions are assumed, and boundary nodes are removed
    from the returned operator.

    Parameters
    ----------
    nx : int
        Number of (interior) grid points in x
    ny : int
        Number of (interior) grid points in x
    nu : float
        Scalar diffusion coefficient
    v : function
        Velocity function of x and y, evaluated at each grid point.
        Should return an (N, 2) array describing the velocity field.
    bc : function
        Function to evaluate at the boundary conditions,
        if None is given then homogeneous bc are assumed.

    Returns
    -------
    Grid object
    """
    dx = 1./(nx + 1)
    dy = 1./(ny + 1)

    x = np.linspace(0, 1, nx + 2)
    y = np.linspace(0, 1, ny + 2)
    xx, yy = np.meshgrid(x, y)

    # Diffusion term

    Ax = (nu * (1./dx**2) * (sp.eye(nx+2) * 2 - sp.eye(nx+2, k=-1) - sp.eye(nx+2, k=1)))
    Ay = (nu * (1./dy**2) * (sp.eye(ny+2) * 2 - sp.eye(ny+2, k=-1) - sp.eye(ny+2, k=1)))

    A = sp.kron(Ax, sp.eye(ny + 2)) + sp.kron(sp.eye(nx + 2), Ay)

    # Discretize advection term as div(vu) = v_x du/dx + v_y du/dy
    # using centered differences

    Cx = (2/dx) * (sp.eye(nx + 2, k=1) - sp.eye(nx + 2, k=-1))
    Cy = (2/dy) * (sp.eye(ny + 2, k=1) - sp.eye(ny + 2, k=-1))

    v_eval = v(xx.flatten(), yy.flatten())

    C = ((sp.diags(v_eval[:, 1]) @ sp.kron(Cx, sp.eye(ny + 2))) +
         (sp.diags(v_eval[:, 0]) @ sp.kron(sp.eye(nx + 2), Cy)))

    A, b = _eval_bc(A + C, nx, ny, bc)

    return Grid(A,
                shape=(nx, ny),
                xlim=(x[1], x[-1]),
                ylim=(y[1], y[-1]),
                bc=BoundaryCondition.DIRICHLET), b


class Stencil:
    """Base class of a Stencil."""

    width = 3

    def _global_pos_to_idx(self, pos):
        return np.sum(pos * self.strides)

    def _stencil_idx_to_pos(self, idx):
        indices = []
        for dim in range(self.ndim - 1, -1, -1):
            stride = Stencil.width ** dim
            component = int(idx / stride)
            indices.append(component)
            idx -= component * stride
        return np.array(indices)[::-1]

    def _idx_in_bounds(self, i):
        for dim in range(self.ndim):
            idx = i[dim]
            if idx < 0 or idx >= self.shape[dim]:
                return False
        return True

    def __init__(self, grid, *indices):
        """Initialize stencil entries."""
        self.grid = grid
        self.shape = self.grid.shape
        self.pos = np.array(indices)
        self.ndim = len(self.shape)

        assert self.ndim in (2, 3)

        self.stencil_values = np.zeros((Stencil.width,) * self.ndim)
        self.stencil_indices = np.zeros((Stencil.width,) * self.ndim,
                                        dtype=np.int64)

        self.strides = np.cumprod(np.insert(np.array(self.shape), 0, 1))[:-1]
        self.stencil_numel = Stencil.width ** self.ndim
        self.row_idx = self._global_pos_to_idx(self.pos)

        def update_val(stencil_pos, global_pos):
            if self._idx_in_bounds(global_pos):
                global_idx = self._global_pos_to_idx(global_pos)
                try:
                    v = self.grid.operator[self.row_idx, global_idx]
                except IndexError:
                    v = 0.
                self.stencil_values[stencil_pos] = v
                self.stencil_indices[stencil_pos] = global_idx
            else:
                self.stencil_values[stencil_pos] = 0.
                self.stencil_indices[stencil_pos] = -1

        if self.ndim == 2:
            i, j = self.pos

            update_val((1, 1), (i, j))          # O
            update_val((1, 2), (i, j + 1))      # N
            update_val((2, 2), (i + 1, j + 1))  # NE
            update_val((2, 1), (i + 1, j))      # E
            update_val((2, 0), (i + 1, j - 1))  # SE
            update_val((1, 0), (i, j - 1))      # S
            update_val((0, 0), (i - 1, j - 1))  # SW
            update_val((0, 1), (i - 1, j))      # W
            update_val((0, 2), (i - 1, j + 1))  # NW
        else:
            raise NotImplementedError('3D is not implemented.')

    def __getitem__(self, key):
        """Return entries of a stencil."""
        assert isinstance(key, tuple)
        new_key = np.array(key) + 1  # Shift indices into corrcet range
        return self.stencil_values[tuple(new_key)]

    def __setitem__(self, key, val):
        """Set entries of a stencil."""
        assert isinstance(key, tuple)
        new_key = np.array(key) + 1
        self.grid.operator[self.row_idx, self.stencil_indices[new_key]] = val

    def __repr__(self):
        """Return representation of stencil."""
        out = np.zeros((3, 3))
        for x in range(-1, 2):
            for y in range(-1, 2):
                out[1 - y, x + 1] = self.__getitem__((x, y))
        return repr(out)
