import numpy as np

from src.grid import make_stock_grid
from src.payoff import call_payoff, put_payoff
from src.boundary_conditions import call_boundaries, put_boundaries


def stability_limit(S_max, K, r, sigma, T, M):
    """
    Maximum number of time steps N required for explicit stability.

    The explicit scheme is stable only when dt <= 1 / (sigma^2*(M-1)^2 + r).
    Returns the minimum N such that dt = T/N satisfies this bound.
    """
    dt_max = 1.0 / (sigma**2 * (M - 1)**2 + r)
    return int(np.ceil(T / dt_max))


def solve_explicit(S_max, K, r, sigma, T, M, N, option_type="call"):
    """
    Price a European option with the fully explicit finite difference method.

    Each time step is a simple matrix-vector multiply — no linear solve needed.
    Conditionally stable: requires dt <= 1 / (sigma^2*(M-1)^2 + r).

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

    # explicit coefficients: V^{n+1}_i = alpha*V_{i-1} + beta*V_i + gamma*V_{i+1}
    alpha = 0.5 * dt * (sigma**2 * i**2 - r * i)
    beta  = 1.0 - dt * (sigma**2 * i**2 + r)
    gamma = 0.5 * dt * (sigma**2 * i**2 + r * i)

    # warn if scheme is unstable
    if np.any(beta < 0):
        import warnings
        worst = dt * (sigma**2 * (M - 1)**2 + r)
        warnings.warn(
            f"Explicit scheme is UNSTABLE (dt too large). "
            f"Stability requires dt*(sigma^2*(M-1)^2 + r) <= 1, "
            f"got {worst:.3f}. Use N >= {stability_limit(S_max, K, r, sigma, T, M)}."
        )

    # march forward in tau from 0 to T
    for n in range(N):
        tau_new = (n + 1) * dt
        lower_bc, upper_bc = boundary_fn(S_max, K, r, tau_new)

        V_new = np.empty_like(V)
        V_new[1:M] = alpha * V[0:M-1] + beta * V[1:M] + gamma * V[2:M+1]
        V_new[0]   = lower_bc
        V_new[M]   = upper_bc

        V = V_new

    return S, V
