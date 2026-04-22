import numpy as np

from src.core import OptionParams, GridParams, make_stock_grid, payoff, boundary_values


def solve_explicit(S_max, K, r, sigma, T, M, N, option_type="call"):
    """
    Price a European option with the explicit finite difference method.

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
        raise ValueError("option_type must be 'call' or 'put'")

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

    # interior node indices i = 1,...,M-1
    i = np.arange(1, M)

    a = 0.5 * dt * (sigma**2 * i**2 - r * i)
    b = 1.0 - dt * (sigma**2 * i**2 + r)
    c = 0.5 * dt * (sigma**2 * i**2 + r * i)

    # step backward from t = T to t = 0
    for n in range(N - 1, -1, -1):
        t = n * dt
        left_bc, right_bc = boundary_values(t, opt, grid)

        V_new = np.zeros_like(V)
        V_new[0] = left_bc
        V_new[M] = right_bc

        V_new[1:M] = a * V[0:M-1] + b * V[1:M] + c * V[2:M+1]

        V = V_new

    return S, V

def stability_limit(T, r, sigma, M):
    """
    Minimum N such that the explicit scheme satisfies

        dt * (sigma^2 * (M-1)^2 + r) <= 1

    with dt = T / N.
    """
    if M < 2:
        raise ValueError("M >= 2")
    if T < 0:
        raise ValueError("T >= 0")
    if sigma < 0:
        raise ValueError("sigma >= 0")

    dt_max = 1.0 / (sigma**2 * (M - 1)**2 + r)
    return int(np.ceil(T / dt_max))