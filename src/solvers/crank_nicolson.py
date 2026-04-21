import numpy as np
from scipy.linalg import solve_banded

from src.grid import make_stock_grid
from src.payoff import call_payoff, put_payoff
from src.boundary_conditions import call_boundaries, put_boundaries


def solve_crank_nicolson(S_max, K, r, sigma, T, M, N, option_type="call"):
    """
    Price a European option with the Crank-Nicolson finite difference method.

    CN averages the implicit and explicit operators 50/50, giving second-order
    accuracy in both time (dt^2) and space (dS^2) vs first-order for implicit.

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

    # LHS coefficients (implicit half): A * V^{n+1} = rhs
    alpha_lhs = -0.25 * dt * (sigma**2 * i**2 - r * i)
    beta_lhs  =  1.0 + 0.5 * dt * (sigma**2 * i**2 + r)
    gamma_lhs = -0.25 * dt * (sigma**2 * i**2 + r * i)

    # RHS coefficients (explicit half): applied to V^n
    alpha_rhs =  0.25 * dt * (sigma**2 * i**2 - r * i)
    beta_rhs  =  1.0 - 0.5 * dt * (sigma**2 * i**2 + r)
    gamma_rhs =  0.25 * dt * (sigma**2 * i**2 + r * i)

    # LHS banded matrix (constant across time steps)
    ab = np.zeros((3, M - 1))
    ab[0, 1:]  = gamma_lhs[:-1]
    ab[1, :]   = beta_lhs
    ab[2, :-1] = alpha_lhs[1:]

    # march forward in tau from 0 to T
    for n in range(N):
        tau_old = n * dt
        tau_new = (n + 1) * dt

        lower_old, upper_old = boundary_fn(S_max, K, r, tau_old)
        lower_new, upper_new = boundary_fn(S_max, K, r, tau_new)

        # Apply explicit operator B to interior of V^n
        rhs = beta_rhs * V[1:M]
        rhs[1:]  += alpha_rhs[1:]  * V[1:M-1]   # sub-diagonal: V[i-1] for i=2..M-1
        rhs[:-1] += gamma_rhs[:-1] * V[2:M]     # super-diagonal: V[i+1] for i=1..M-2

        # Add boundary contributions from old time level (explicit side)
        rhs[0]  += alpha_rhs[0]  * lower_old
        rhs[-1] += gamma_rhs[-1] * upper_old

        # Subtract boundary contributions from new time level (implicit side)
        rhs[0]  -= alpha_lhs[0]  * lower_new
        rhs[-1] -= gamma_lhs[-1] * upper_new

        V[1:M] = solve_banded((1, 1), ab, rhs)
        V[0]   = lower_new
        V[M]   = upper_new

    return S, V
