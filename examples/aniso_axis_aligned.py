"""Axis-aligned anisotropy test: alpha u_xx + beta u_yy = f.

This example demonstrates that line relaxation should be aligned with the
strong coupling direction:

- If alpha >> beta (strong coupling in x), use x-line relaxation ('line_x')
- If beta >> alpha (strong coupling in y), use y-line relaxation ('line_y')

It also compares against pointwise smoothing ('point').
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import boxpy


def rhs(x, y):
    # Smooth forcing (keeps the solution well-behaved)
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def run_case(n, alpha, beta, smoother_name, maxiter=30):
    grid, _ = boxpy.grid.create_diffusion_dirichlet_2d(n, n, np.diag([alpha, beta]))

    ml = boxpy.boxmg_solver(
        grid,
        symmetric=True,
        presmoother=(smoother_name, {"iterations": 1, "cycling_down": True}),
        postsmoother=(smoother_name, {"iterations": 1, "cycling_down": False}),
        max_levels=10,
    )

    x0 = grid.interp_fcn(lambda x, y: np.random.normal(size=x.shape)).flatten()
    x0 = x0 / la.norm(x0)
    b = grid.interp_fcn(rhs).flatten()

    residuals = []
    x = ml.solve(b, x0=x0, residuals=residuals, tol=1e-10, maxiter=maxiter)

    residuals = np.asarray(residuals, dtype=float)
    conv = residuals[1:] / residuals[:-1]
    return x, residuals, conv

n = 128

cases = [
    {'alpha': 1.0, 'beta': 1e-3},
    {'alpha': 1e-3, 'beta': 1.0},
]

fig, axes = plt.subplots(len(cases), 2, figsize=(10, 6), constrained_layout=True)

for row, case in enumerate(cases):
    alpha = case['alpha']
    beta = case['beta']

    for smoother_name in ('point', 'line_x', 'line_y'):
        _x, res, conv = run_case(n, alpha, beta, smoother_name)
        label = f'{smoother_name}'

        ax_res = axes[row, 0]
        ax_conv = axes[row, 1]
        ax_res.semilogy(res, 'o-', markersize=3, label=label)
        ax_conv.plot(np.arange(1, len(res)), conv, '--', label=label)

    axes[row, 0].grid(True)
    axes[row, 1].grid(True)

    axes[row, 0].set_xlabel('Multigrid Iteration')
    axes[row, 0].set_ylabel('Absolute Residual')
    axes[row, 1].set_xlabel('Multigrid Iteration')
    axes[row, 1].set_ylabel('Convergence Factor')

    axes[row, 0].set_title(f'alpha={alpha:g}, beta={beta:g}')
    axes[row, 1].set_title(f'alpha={alpha:g}, beta={beta:g}')

    axes[row, 0].legend(loc=0)
    axes[row, 1].legend(loc=0)

fig.suptitle(r'Anisotropic problem: $\alpha u_{xx} + \beta u_{yy} = f$')
plt.show()