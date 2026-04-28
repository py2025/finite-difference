import numpy as np
from scipy.linalg import solve_banded

from ..core import OptionParams, GridParams, make_stock_grid, payoff, boundary_values

def solve_american(S_max, K, r, sigma, T, M, N, option_type="put"):
    """
    Price an American option using Crank-Nicolson + early exercise constraint.

    At each time step, after solving the linear system, we enforce:
        V = max(V, intrinsic_value)

    This is the free-boundary condition that defines American options.

    Returns
    -------
    S : ndarray
    V : ndarray
    """
    if M < 2 or N < 1:
        raise ValueError("Require M >= 2 and N >= 1")
    if option_type not in ("call", "put"):
        raise ValueError("option_type: 'call' or 'put'")

    opt = OptionParams(
        S0=0.0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type=option_type,
        exercise_type="american",
    )
    grid = GridParams(S_max=S_max, M=M, N=N)

    S = make_stock_grid(grid)
    dt = T / N

    V = payoff(S, K, option_type)

    i = np.arange(1, M)

    alpha = 0.25 * dt * (sigma**2 * i**2 - r * i)
    beta  = -0.5 * dt * (sigma**2 * i**2 + r)
    gamma = 0.25 * dt * (sigma**2 * i**2 + r * i)

    ab = np.zeros((3, M - 1))
    ab[0, 1:]  = -gamma[:-1]
    ab[1, :]   = 1.0 - beta
    ab[2, :-1] = -alpha[1:]

    intrinsic = payoff(S[1:M], K, option_type)

    for n in range(N - 1, -1, -1):
        t = n * dt
        left_bc, right_bc = boundary_values(t, opt, grid)

        rhs = (
            alpha * V[0:M-1]
            + (1.0 + beta) * V[1:M]
            + gamma * V[2:M+1]
        )
        rhs[0]  += alpha[0] * left_bc
        rhs[-1] += gamma[-1] * right_bc

        V[1:M] = solve_banded((1, 1), ab, rhs)
        V[1:M] = np.maximum(V[1:M], intrinsic)  # early exercise constraint
        V[0] = left_bc
        V[M] = right_bc

    return S, V


def binomial_american(S0, K, r, sigma, T, N, option_type="put"):
    """
    Cox-Ross-Rubinstein binomial tree pricer for American options.
    Used as an independent benchmark to validate solve_american.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # terminal stock prices
    j = np.arange(N + 1)
    ST = S0 * u ** (N - 2 * j)

    V = payoff(ST, K, option_type)

    # step backward
    for n in range(N - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        j = np.arange(n + 1)
        ST = S0 * u ** (n - 2 * j)
        V = np.maximum(V, payoff(ST, K, option_type))

    return float(V[0])
