"""Show images of BoxMG solutions."""
import numpy as np
import matplotlib.pyplot as plt
import boxpy
import boxpy.interpolation

N = 64

grid, _ = boxpy.grid.create_poisson_dirichlet_2d(N, N, 1.0)
ml = boxpy.boxmg_solver(grid)

print(ml)

N_lvl = len(ml.levels)
fig, axs = plt.subplots(1, len(ml.levels))

img = grid.interp_fcn(lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)).flatten()

for i in range(0, N_lvl - 1):
    level = ml.levels[i]
    axs[i].imshow(img.reshape(level.grid.shape))
    img = level.R @ img

level = ml.levels[-1]
axs[-1].imshow(img.reshape(level.grid.shape))

plt.show(block=True)
