import numpy as np
from scipy.linalg import solve_banded

from ..core import OptionParams, GridParams, make_stock_grid, payoff, boundary_values

def solve_crank_nicolson(S_max, K, r, sigma, T, M, N, option_type="call"):
    """
    Price a European option with the Crank-Nicolson finite difference method.

    Returns
    -------
    S : ndarray
        Stock price grid
    V : ndarray
        Option values at t = 0
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
        exercise_type="european",
    )
    grid = GridParams(S_max=S_max, M=M, N=N)

    S = make_stock_grid(grid)
    dt = T / N

    # terminal condition V(S,T)
    V = payoff(S, K, option_type)

    # interior nodes i = 1,...,M-1
    i = np.arange(1, M)

    alpha = 0.25 * dt * (sigma**2 * i**2 - r * i)
    beta  = -0.5 * dt * (sigma**2 * i**2 + r)
    gamma = 0.25 * dt * (sigma**2 * i**2 + r * i)

    # Left-hand side matrix: (I - 0.5*dt*L)
    ab = np.zeros((3, M - 1))
    ab[0, 1:] = -gamma[:-1]      # superdiagonal
    ab[1, :]  = 1.0 - beta       # diagonal
    ab[2, :-1] = -alpha[1:]      # subdiagonal

    # step backward from t = T to t = 0
    for n in range(N - 1, -1, -1):
        t = n * dt
        left_bc, right_bc = boundary_values(t, opt, grid)

        rhs = (
            alpha * V[0:M-1]
            + (1.0 + beta) * V[1:M]
            + gamma * V[2:M+1]
        )

        # boundary adjustments
        rhs[0] += alpha[0] * left_bc
        rhs[-1] += gamma[-1] * right_bc

        V[1:M] = solve_banded((1, 1), ab, rhs)
        V[0] = left_bc
        V[M] = right_bc

    return S, V