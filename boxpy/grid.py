"""Grid and boundary condition classes."""

import enum
import numpy as np
import scipy.sparse as sp


class BoundaryCondition(enum.Enum):
    """Enumerate boundary conditions."""

    DIRICHLET = 1
    PERIODIC = 2


class Grid:
    """Based Grid class."""

    # pylint: disable-next=too-many-arguments
    def __init__(self, operator, shape, xlim, ylim, bc):
        """Initialize the grid class."""
        self.operator = operator
        self.shape = shape
        self.ndim = len(self.shape)
        self.xlim = xlim
        self.ylim = ylim
        self.bc = bc

    def create_poisson_dirichlet_2d(self, nx, ny, nu):
        """Construct a Poisson operator in 2D.

        Parameters
        ----------
        nx : int
            Number of grid points in x
        nx : int
            Number of grid points in x
        nu : float
            Diffusion coefficient

        Return
        ------
        Grid object
        """
        dx = 1./(nx + 1)
        dy = 1./(ny + 1)

        Ax = (nu * (1./dx**2) * (sp.eye(nx) * 2 - sp.eye(nx, k=-1) - sp.eye(nx, k=1)))
        Ay = (nu * (1./dy**2) * (sp.eye(ny) * 2 - sp.eye(ny, k=-1) - sp.eye(ny, k=1)))

        A = sp.kron(Ax, sp.eye(ny)) + sp.kron(sp.eye(nx), Ay)

        x = np.linspace(0, 1, nx + 2)
        y = np.linspace(0, 1, ny + 2)

        return Grid(A,
                    shape=(nx, ny),
                    xlim=(x[1], x[-1]),
                    ylim=(y[1], y[-1]),
                    bc=BoundaryCondition.DIRICHLET)

    def coarsen(self, new_shape, P, R=None):
        """Coarsen the operator."""
        AH = R @ self.operator @ P
        return Grid(AH, new_shape, self.xlim, self.ylim, self.bc)

    def __call__(self, *indices):
        """Retrieve stencils."""
        return Stencil(self, *(np.array(indices) - 1))

    def __getitem__(self, indices):
        """Retreive a stencil."""
        return Stencil(self, *indices)

    def __setitem__(self, indices, stencil_val):
        """Set stencils."""
        stencil = Stencil(self, *indices)
        stencil[:] = stencil_val

    def interp_fcn(self, f):
        """Interpolate a function.

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
        new_key = np.array(key) + 1
        return self.stencil_values[new_key]

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
