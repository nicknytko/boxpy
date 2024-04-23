"""Solve an anisotropic diffusion problem."""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import boxpy
import boxpy.interpolation
import cProfile
import sys

N = 64

# Set up a diffusion problem with weak diffusion along the angle pi/4 radians
theta = np.pi / 4
epsilon = 1e-3

Q = np.array([
    [ np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)]
])
D = Q @ np.diag([1., epsilon]) @ Q.T

# Sinusoidal-y forcing term

def rhs(x, y):
    return np.sin(2*np.pi*x) * np.sin(2*np.pi*y)

grid, _ = boxpy.grid.create_diffusion_dirichlet_2d(N, N, D)
ml = boxpy.boxmg_solver(grid)

print(ml)

x0 = grid.interp_fcn(lambda x, y: np.random.normal(size=x.shape)).flatten()
x0 = x0 / la.norm(x0)

b = grid.interp_fcn(lambda x, y: rhs(x, y)).flatten()

res = []
x = ml.solve(b, x0, residuals=res)
res = np.array(res)

conv = res[1:] / res[:-1]

fig = plt.figure()
ax = plt.gca()

resline = ax.semilogy(res, 'o-', markersize=3, label='Residual')
ax.grid()
ax.set_xlabel('Multigrid Iteration')
ax.set_ylabel('Absolute Residual')

ax2 = ax.twinx()
convline = ax2.plot(np.arange(1, len(res)), conv,
                    linestyle='--', color='tab:orange', label='Convergence')
ax2.set_ylabel('Convergence Factor')

lines = resline + convline
ax2.legend(lines, [line.get_label() for line in lines], loc=0)

fig.suptitle('BoxMG Residual History')

plt.figure()
plt.imshow(x.reshape((N, N)))
plt.colorbar()
plt.title('Numerical solution')

plt.show()
