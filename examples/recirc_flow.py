"""Solve the recirculating flow problem from
Finite Elements and Fast Iterative Solvers.


"""
import numpy as np
import numpy.linalg as la
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import boxpy
import boxpy.interpolation

N = 64

# Define problem

def bc(x, y):
    return (x == 1).astype(np.float64)

def v(x, y):
    x = x * 2 - 1
    y = y * 2 - 1
    return np.column_stack((
        2. * y * (1 - x ** 2),
        -2 * x * (1 - y ** 2)
    ))

eps = 0.1
grid, b = boxpy.grid.create_advection_dirichlet_2d(N, N, eps, v, bc)
ml = boxpy.boxmg_solver(grid)

print(ml)

# Solve

x0 = grid.interp_fcn(lambda x, y: np.random.normal(size=x.shape)).flatten()
x0 = x0 / la.norm(x0)

res = []
x = ml.solve(b, x0, residuals=res)
res = np.array(res)

# Plot convergence history

conv = res[1:] / res[:-1]
fig = plt.figure()
ax = plt.gca()
resline = ax.semilogy(res / la.norm(b), 'o-', markersize=3, label='Residual')
ax.grid()
ax.set_xlabel('Multigrid Iteration')
ax.set_ylabel('Relative Residual')

ax2 = ax.twinx()
convline = ax2.plot(np.arange(1, len(res)), conv,
                    linestyle='--', color='tab:orange', label='Convergence')
ax2.set_ylabel('Convergence Factor')

lines = resline + convline
ax2.legend(lines, [line.get_label() for line in lines], loc=0)

fig.suptitle('BoxMG Residual History')

# Plot solution

plt.figure()
plt.imshow(x.reshape((N, N))[::-1,:])
plt.colorbar()
plt.title('Solution')

plt.show()
