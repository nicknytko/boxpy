import numpy as np
import scipy.sparse as sp


class Grid:
    def __init__(self, op, shape):
        self.op = op
        self.shape = shape
        self.ndim = len(self.shape)

    def create_poisson_dirichlet_2d(Nx, Ny, nu):
        dx = 1./(Nx + 1)
        dy = 1./(Ny + 1)

        Ax = nu * (1./dx**2) * (sp.eye(Nx) * 2 - sp.eye(Nx, k=-1) - sp.eye(Nx, k=1))
        Ay = nu * (1./dy**2) * (sp.eye(Ny) * 2 - sp.eye(Ny, k=-1) - sp.eye(Ny, k=1))

        A = sp.kron(Ax, sp.eye(Ny)) + sp.kron(sp.eye(Nx), Ay)
        return Grid(A, (Nx, Ny))


class Stencil:
    WIDTH = 3

    def _pos_to_idx(self, i):
        return np.sum(self.strides * i)


    def _idx_to_pos(self, i):
        indices = []
        for stride in self.strides:
            indices.append(int(i / stride))
            i -= indices[-1] * stride
        return np.array(indices)


    def _stencil_idx_to_pos(self, i):
        indices = []
        for dim in range(self.ndim - 1, -1, -1):
            stride = Stencil.WIDTH ** dim
            idx = int(i / stride)
            indices.append(idx)
            i -= idx * stride
        return np.array(indices)


    def _idx_in_bounds(self, i):
        for dim in range(self.ndim):
            idx = i[dim]
            if idx < 0 or idx >= self.shape[dim]:
                return False
        return True


    def __init__(self, grid, *indices):
        self.grid = grid
        self.shape = self.grid.shape
        self.idx = np.array(indices)
        self.ndim = len(self.shape)

        self.stencil_values = np.zeros((Stencil.WIDTH,) * self.ndim)
        self.stencil_indices = np.zeros((Stencil.WIDTH,) * self.ndim, dtype=np.int64)

        self.strides = np.cumprod(np.insert(np.array(self.shape), 0, 1))[::-1][1:]
        self.stencil_numel = Stencil.WIDTH ** self.ndim
        self.row_idx = self._pos_to_idx(self.idx)

        for i in range(self.stencil_numel):
            stencil_pos = self._stencil_idx_to_pos(i)
            matrix_pos = stencil_pos + self.idx - 1
            if self._idx_in_bounds(matrix_pos):
                matrix_idx = self._pos_to_idx(matrix_pos)
                self.stencil_indices[*stencil_pos] = matrix_idx
                try:
                    self.stencil_values[*stencil_pos] = self.grid.op[self.row_idx, matrix_idx]
                except:
                    self.stencil_values[*stencil_pos] = 0.
            else:
                self.stencil_indices[*stencil_pos] = -1
                self.stencil_values[*stencil_pos] = np.nan


    def _transform_slicing_indices(self, key):
        assert(isinstance(key, tuple))
        assert(len(key) == self.ndim)

        new_key = []
        for dim in range(self.ndim, -1, -1):
            key_item = key[dim]

            if isinstance(key_item, int):
                new_key.append(key_item + 1)
            elif isinstance(key_item, slice):
                new_indices = [None, None, key_item.step]
                if key_item.start is not None:
                    new_indices[0] = min(1 + key_item.start, Stencil.WIDTH)
                if key_item.stop is not None:
                    new_indices[1] = min(1 + key_item.stop, Stencil.WIDTH)
                new_key.append(slice(*new_indices))
        return tuple(new_key)


    def __getitem__(self, key):
        new_key = self._transform_slicing_indices(key)
        return self.stencil_values[*new_key]


    def __setitem__(self, key, val):
        new_key = self._transform_slicing_indices(key)
        indices = self.stencil_indices[*new_key]
        self.grid.op[self.row_idx, indices] = val
