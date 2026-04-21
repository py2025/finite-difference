import numpy as np
from scipy.linalg import solve_banded

from src.grid import make_stock_grid
from src.payoff import call_payoff, put_payoff
from src.boundary_conditions import call_boundaries, put_boundaries


def solve_implicit(S_max, K, r, sigma, T, M, N, option_type="call"):
    """
    Price a European option with the fully implicit finite difference method.

    Returns
    -------
    S : ndarray
        Stock price grid
    V : ndarray
        Option values at t = 0
    """
    if M < 2 or N < 1:
        raise ValueError("Require M >= 2 and N >= 1")

    S = make_stock_grid(S_max, M)
    dt = T / N

    if option_type == "call":
        payoff_fn = call_payoff
        boundary_fn = call_boundaries
    elif option_type == "put":
        payoff_fn = put_payoff
        boundary_fn = put_boundaries
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # terminal condition at tau = 0
    V = payoff_fn(S, K)

    # interior nodes i = 1,...,M-1
    i = np.arange(1, M)

    alpha = -0.5 * dt * (sigma**2 * i**2 - r * i)
    beta  = 1.0 + dt * (sigma**2 * i**2 + r)
    gamma = -0.5 * dt * (sigma**2 * i**2 + r * i)

    # banded matrix for solve_banded
    ab = np.zeros((3, M - 1))
    ab[0, 1:] = gamma[:-1]
    ab[1, :]  = beta
    ab[2, :-1] = alpha[1:]

    # march forward in tau from 0 to T
    for n in range(N):
        tau = (n + 1) * dt

        lower_bc, upper_bc = boundary_fn(S_max, K, r, tau)

        rhs = V[1:M].copy()
        rhs[0]  -= alpha[0] * lower_bc
        rhs[-1] -= gamma[-1] * upper_bc

        V[1:M] = solve_banded((1, 1), ab, rhs)
        V[0] = lower_bc
        V[M] = upper_bc

    return S, V