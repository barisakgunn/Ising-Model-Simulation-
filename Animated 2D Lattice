import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
n = 10                 # lattice size
kB = 0.007             # Boltzmann constant
steps_per_frame = 500  # how many MC updates before each frame

# Initialize random spin lattice (-1 or +1)
lattice = np.random.choice([-1, 1], size=(n, n))

# Temperature input
T = float(input("Choose an arbitrary positive Temperature value: "))
J = 1.0 / T

# --- Monte Carlo step function ---
def monte_carlo_step(lattice, T, J, kB, steps=1):
    n = lattice.shape[0]
    for _ in range(steps):
        x, y = np.random.randint(0, n, size=2)
        s = lattice[x, y]

        # Nearest neighbors (periodic boundary conditions)
        xa, xb = (x + 1) % n, (x - 1) % n
        ya, yb = (y + 1) % n, (y - 1) % n

        neighbors_sum = (
            lattice[xa, y] + lattice[xb, y] +
            lattice[x, ya] + lattice[x, yb]
        )

        dE = 2 * J * s * neighbors_sum

        # Metropolis acceptance rule
        if dE <= 0 or np.random.rand() < np.exp(-dE / (kB * T)):
            lattice[x, y] = -s
    return lattice

# --- Setup plot ---
fig, ax = plt.subplots()
scatter = ax.scatter([], [], s=100)
ax.set_xlim(-1, n)
ax.set_ylim(-1, n)
ax.set_aspect("equal")
ax.set_title("2D Ising Model (Spin Glass Animation)")

# --- Animation update function ---
def update(frame):
    global lattice
    lattice = monte_carlo_step(lattice, T, J, kB, steps=steps_per_frame)

    xs, ys = np.indices((n, n))
    xs, ys = xs.flatten(), ys.flatten()
    spins = lattice.flatten()

    colors = ["red" if s == 1 else "blue" for s in spins]
    markers = ["^" if s == 1 else "v" for s in spins]

    ax.clear()
    for x, y, s, c, m in zip(xs, ys, spins, colors, markers):
        ax.scatter(x, y, c=c, marker=m, s=100)

    ax.set_xlim(-1, n)
    ax.set_ylim(-1, n)
    ax.set_aspect("equal")
    ax.set_title(f"2D Ising Model (Frame {frame})")

# --- Run animation ---
ani = FuncAnimation(fig, update, frames=200, interval=200, repeat=False)
plt.show()
