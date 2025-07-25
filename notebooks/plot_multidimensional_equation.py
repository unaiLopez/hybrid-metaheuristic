import numpy as np
import matplotlib.pyplot as plt

# 3‑D Ackley (vectorized)
def ackley_3d(x, y, z, a=20, b=0.2, c=2*np.pi):
    sum_sq  = x**2 + y**2 + z**2
    sum_cos = np.cos(c*x) + np.cos(c*y) + np.cos(c*z)
    n = 3.0
    term1 = -a * np.exp(-b * np.sqrt(sum_sq/n))
    term2 = -np.exp(    sum_cos/n)
    return term1 + term2 + a + np.exp(1)

# Your solution and global optimum
my_solution    = [-4.1,  7.1, 6.9]
global_optimum = [ 0.0,  0.0,  0.0]

# Fix x2 at your solution's z
z_fixed = my_solution[2]

# Build a grid over x0,x1
grid_size = 150
x = np.linspace(-10, 10, grid_size)
y = np.linspace(-10, 10, grid_size)
X, Y = np.meshgrid(x, y)

# Evaluate Ackley on that grid slice
Z = ackley_3d(X, Y, z_fixed)

# Evaluate your points
f_my_solution = ackley_3d(*my_solution)
f_opt_fixed   = ackley_3d(global_optimum[0], global_optimum[1], z_fixed)

# Plot
fig = plt.figure(figsize=(12,9))
ax  = fig.add_subplot(111, projection='3d')

# 1) Surface plotted first, slightly translucent
surf = ax.plot_surface(
    X, Y, Z,
    cmap='viridis',
    edgecolor='black',
    linewidth=0.3,
    alpha=0.8      # make it partially see‑through
)

# 2) Then scatter your solution & optimum, with depthshade off so they stay bright
ax.scatter(
    my_solution[0], my_solution[1], f_my_solution,
    color='red',
    marker='X',
    s=500,
    edgecolor='k',
    linewidth=1,
    depthshade=False,
    label=f'Your solution\nf={f_my_solution:.2f}'
)

ax.scatter(
    global_optimum[0], global_optimum[1], f_opt_fixed,
    color='lime',
    marker='X',
    s=500,
    edgecolor='k',
    linewidth=1,
    depthshade=False,
    label=f'Global optimum slice\nf={f_opt_fixed:.2f}'
)

# Labels, title, legend
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('Ackley value')
ax.set_title(f'Ackley Surface at x2 = {z_fixed}')
ax.legend(loc='upper right')

# Optional: tweak the viewing angle so you can see the points
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()


# 2‑D Ackley
def ackley_2d(x, y, a=20, b=0.2, c=2*np.pi):
    sum_sq  = x**2 + y**2
    sum_cos = np.cos(c*x) + np.cos(c*y)
    n = 2.0
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(    sum_cos / n)
    return term1 + term2 + a + np.exp(1)

# Your 2D solution and the global optimum
sol_2d = (-4.1, 7.1)
opt_2d = (0.0,  0.0)

# Build grid
grid_size = 500
x = np.linspace(-10, 10, grid_size)
y = np.linspace(-10, 10, grid_size)
X, Y = np.meshgrid(x, y)
Z = ackley_2d(X, Y)

# Evaluate points
f_sol = ackley_2d(*sol_2d)
f_opt = ackley_2d(*opt_2d)

# Plot
plt.figure(figsize=(8,6))
# filled contour
cnt = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(cnt, label='f(x,y)')

# scatter solution & optimum
plt.scatter(*sol_2d, c='red',   s=100, marker='X', label=f'Solution ({sol_2d[0]}, {sol_2d[1]})\nf={f_sol:.2f}')
plt.scatter(*opt_2d, c='lime',  s=100, marker='X', label=f'Global optimum\n(0,0)\nf={f_opt:.2f}')

plt.xlabel('x₀')
plt.ylabel('x₁')
plt.title('Ackley Function in 2D')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
