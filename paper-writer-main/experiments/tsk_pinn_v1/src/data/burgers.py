import numpy as np


def sample_training_points(n_f=2000, n_b=200, n_i=200, seed=0):
    rng = np.random.default_rng(seed)

    x_f = rng.uniform(-1, 1, size=(n_f, 1))
    t_f = rng.uniform(0, 1, size=(n_f, 1))

    t_b = rng.uniform(0, 1, size=(n_b, 1))
    x_b0 = -np.ones((n_b // 2, 1))
    x_b1 = np.ones((n_b - n_b // 2, 1))
    x_b = np.vstack([x_b0, x_b1])
    t_b = t_b[: x_b.shape[0]]
    u_b = np.zeros_like(x_b)

    x_i = rng.uniform(-1, 1, size=(n_i, 1))
    t_i = np.zeros((n_i, 1))
    u_i = -np.sin(np.pi * x_i)

    return {
        "xf": np.hstack([x_f, t_f]),
        "xb": np.hstack([x_b, t_b]),
        "ub": u_b,
        "xi": np.hstack([x_i, t_i]),
        "ui": u_i,
    }


def fd_reference(nx=256, nt=2000, nu=0.01 / np.pi):
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u = np.zeros((nt, nx), dtype=np.float64)
    u[0] = -np.sin(np.pi * x)
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    for n in range(nt - 1):
        un = u[n].copy()
        # Upwind scheme for convection term (stable)
        um = un[:-2]   # u_{i-1}
        u0 = un[1:-1]  # u_i
        up = un[2:]    # u_{i+1}
        # Choose upwind direction based on sign of u
        ux = np.where(u0 > 0, (u0 - um) / dx, (up - u0) / dx)
        uxx = (up - 2*u0 + um) / (dx * dx)
        u[n + 1, 1:-1] = u0 + dt * (-u0 * ux + nu * uxx)
        u[n + 1, 0] = 0.0
        u[n + 1, -1] = 0.0
    return x, t, u
