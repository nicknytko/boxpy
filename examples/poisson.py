"""Solve Poisson."""
import cProfile
import sys
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import boxpy
import boxpy.interpolation

N = 128

grid, _ = boxpy.grid.create_poisson_dirichlet_2d(N, N, 1.0)

if '--profile' in sys.argv:
    cProfile.runctx('ml = boxpy.boxmg_symmetric_solver(grid)',
                    globals(), locals(), sort='cumtime')
else:
    ml = boxpy.boxmg_solver(grid)

print(ml)

x0 = grid.interp_fcn(lambda x, y: np.random.normal(size=x.shape)).flatten()
x0 = x0 / la.norm(x0)
b = grid.interp_fcn(lambda x, y: np.zeros_like(x)).flatten()

res = []
x = ml.solve(b, x0, residuals=res, tol=1e-13)
res = np.array(res)

m = 1
conv = (res[m:]/res[:-m])**(1/m)

fig = plt.figure()
ax = plt.gca()

resline = ax.semilogy(res, 'o-', markersize=3, label='Residual')
ax.grid()
ax.set_xlabel('Multigrid Iteration')
ax.set_ylabel('Absolute Residual')

ax2 = ax.twinx()
convline = ax2.plot(np.arange(m, len(conv)+m), conv,
                    linestyle='--', color='tab:orange', label='Convergence')
ax2.set_ylabel('Convergence Factor')

lines = resline + convline
ax2.legend(lines, [line.get_label() for line in lines], loc=0)

fig.suptitle('BoxMG Residual History')
plt.show()
