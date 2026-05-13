"""
Extended Heston ADI solver.

This module prices European options under a Heston-style stochastic volatility
model using a Douglas ADI finite-difference scheme.

The model is

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa(t) (theta(t) - v_t) dt + sigma(t) sqrt(v_t) dW^v_t
    d<W^S, W^v>_t = rho(t) dt

The "extended" part is that kappa, theta, sigma, rho, r, and q may be either
constants or functions of time-to-maturity tau.
"""

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from ..core import (
    Grid2DParams,
    make_stock_grid_2d,
    make_vol_grid_2d,
    ds_2d,
    dv_2d,
    dt_2d,
)

NumberOrFunc = Union[float, Callable[[float], float]]


@dataclass
class ExtendedHestonParams:
    kappa: NumberOrFunc
    theta: NumberOrFunc
    sigma: NumberOrFunc
    rho: NumberOrFunc
    v0: float
    r: NumberOrFunc = 0.0
    q: NumberOrFunc = 0.0


def _value(x: NumberOrFunc, tau: float) -> float:
    """Evaluate a constant or time-dependent parameter."""
    return float(x(tau)) if callable(x) else float(x)


class ExtendedHestonADI:
    """Douglas ADI solver for European options under extended Heston."""

    def __init__(
        self,
        params: ExtendedHestonParams,
        grid: Grid2DParams,
        K: float,
        T: float,
        option_type: str = "call",
    ):
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        if K < 0:
            raise ValueError("K must be nonnegative")
        if T < 0:
            raise ValueError("T must be nonnegative")
        if params.v0 < 0:
            raise ValueError("v0 must be nonnegative")

        self.params = params
        self.grid = grid
        self.K = K
        self.T = T
        self.option_type = option_type

        self.S = make_stock_grid_2d(grid)
        self.v = make_vol_grid_2d(grid)
        self.dS = ds_2d(grid)
        self.dv = dv_2d(grid)
        self.dt = dt_2d(T, grid)

        self.M = grid.M
        self.L = grid.L
        self.N = grid.N

    def _params_at(self, tau: float):
        kappa = _value(self.params.kappa, tau)
        theta = _value(self.params.theta, tau)
        sigma = _value(self.params.sigma, tau)
        rho = _value(self.params.rho, tau)
        r = _value(self.params.r, tau)
        q = _value(self.params.q, tau)

        if kappa < 0:
            raise ValueError("kappa must be nonnegative")
        if theta < 0:
            raise ValueError("theta must be nonnegative")
        if sigma < 0:
            raise ValueError("sigma must be nonnegative")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rho must be in [-1, 1]")

        return kappa, theta, sigma, rho, r, q

    def payoff(self, S):
        S = np.asarray(S, dtype=float)
        if self.option_type == "call":
            return np.maximum(S - self.K, 0.0)
        return np.maximum(self.K - S, 0.0)

    def _apply_bc_tau(self, V, tau):
        """Apply simple far-field boundary conditions."""
        _, _, _, _, r, q = self._params_at(tau)
        df_r = np.exp(-r * tau)
        df_q = np.exp(-q * tau)
        S = self.S

        if self.option_type == "call":
            V[0, :] = 0.0
            V[self.M, :] = S[-1] * df_q - self.K * df_r
            V[:, 0] = np.maximum(S * df_q - self.K * df_r, 0.0)
        else:
            V[0, :] = self.K * df_r
            V[self.M, :] = 0.0
            V[:, 0] = np.maximum(self.K * df_r - S * df_q, 0.0)

        # Neumann at v_max: dV/dv = 0
        V[:, self.L] = V[:, self.L - 1]
        return V

    def _coefficients(self, tau):
        kappa, theta, sigma, rho, r, q = self._params_at(tau)

        S_col = self.S[:, None]
        v_row = self.v[None, :]

        aS = 0.5 * v_row * (S_col ** 2)
        bS = (r - q) * self.S

        aV = 0.5 * sigma ** 2 * self.v
        bV = kappa * (theta - self.v)

        gamma = rho * sigma * v_row * S_col

        return aS, bS, aV, bV, gamma, r

    def _F0(self, V, tau):
        _, _, _, _, gamma, _ = self._coefficients(tau)

        out = np.zeros_like(V)
        out[1:-1, 1:-1] = (
            gamma[1:-1, 1:-1]
            * (V[2:, 2:] - V[2:, :-2] - V[:-2, 2:] + V[:-2, :-2])
            / (4.0 * self.dS * self.dv)
        )
        return out

    def _F1(self, V, tau):
        aS, bS, _, _, _, r = self._coefficients(tau)

        out = np.zeros_like(V)
        a = aS[1:-1, :] / (self.dS ** 2)
        b = bS[1:-1, None] / (2.0 * self.dS)

        out[1:-1, :] = (
            a * (V[2:, :] - 2.0 * V[1:-1, :] + V[:-2, :])
            + b * (V[2:, :] - V[:-2, :])
            - 0.5 * r * V[1:-1, :]
        )
        return out

    def _F2(self, V, tau):
        _, _, aV, bV, _, r = self._coefficients(tau)

        out = np.zeros_like(V)
        c = aV[None, 1:-1] / (self.dv ** 2)
        d = bV[None, 1:-1] / (2.0 * self.dv)

        out[:, 1:-1] = (
            c * (V[:, 2:] - 2.0 * V[:, 1:-1] + V[:, :-2])
            + d * (V[:, 2:] - V[:, :-2])
            - 0.5 * r * V[:, 1:-1]
        )
        return out

    def _solve_S_implicit(self, rhs, theta_dt, tau_new):
        M, L = self.M, self.L
        Y = self._apply_bc_tau(rhs.copy(), tau_new)

        aS, bS, _, _, _, r = self._coefficients(tau_new)

        a = aS[1:M, 1:L] / (self.dS ** 2)
        b = bS[1:M, None] / (2.0 * self.dS)

        sub = -theta_dt * (a - b)
        mid = 1.0 + theta_dt * (2.0 * a + 0.5 * r)
        sup = -theta_dt * (a + b)

        rhs_int = rhs[1:M, 1:L].copy()
        rhs_int[0, :] -= sub[0, :] * Y[0, 1:L]
        rhs_int[-1, :] -= sup[-1, :] * Y[M, 1:L]

        Y[1:M, 1:L] = _thomas_batch(sub, mid, sup, rhs_int)
        return self._apply_bc_tau(Y, tau_new)

    def _solve_v_implicit(self, rhs, theta_dt, tau_new):
        M, L = self.M, self.L
        Y = self._apply_bc_tau(rhs.copy(), tau_new)

        _, _, aV, bV, _, r = self._coefficients(tau_new)

        c = aV[1:L] / (self.dv ** 2)
        d = bV[1:L] / (2.0 * self.dv)

        sub = -theta_dt * (c - d)
        mid = 1.0 + theta_dt * (2.0 * c + 0.5 * r)
        sup = -theta_dt * (c + d)

        # Neumann at v_max: Y[:, L] = Y[:, L-1]
        mid_eff = mid.copy()
        mid_eff[-1] = mid[-1] + sup[-1]

        rhs_int = rhs[:, 1:L].T.copy()
        rhs_int[0, :] -= sub[0] * Y[:, 0]

        sol = _thomas_batch(sub, mid_eff, sup, rhs_int)
        Y[:, 1:L] = sol.T
        Y[:, L] = Y[:, L - 1]

        return self._apply_bc_tau(Y, tau_new)

    def step(self, V, tau_old, tau_new, theta=0.5):
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0, 1], got {theta}")

        dt = tau_new - tau_old
        tau_mid = 0.5 * (tau_old + tau_new)

        F0V = self._F0(V, tau_mid)
        F1V = self._F1(V, tau_mid)
        F2V = self._F2(V, tau_mid)

        Y0 = V + dt * (F0V + F1V + F2V)
        Y0 = self._apply_bc_tau(Y0, tau_new)

        Y1 = self._solve_S_implicit(Y0 - theta * dt * F1V, theta * dt, tau_new)
        V_new = self._solve_v_implicit(Y1 - theta * dt * F2V, theta * dt, tau_new)

        return V_new

    def solve(self, theta=0.5, verbose=False):
        V = np.empty((self.M + 1, self.L + 1))
        V[:, :] = self.payoff(self.S)[:, None]
        V = self._apply_bc_tau(V, 0.0)

        for n in range(self.N):
            tau_old = n * self.dt
            tau_new = (n + 1) * self.dt

            if verbose and (n % max(1, self.N // 10) == 0):
                print(
                    f"  step {n + 1}/{self.N} "
                    f"tau={tau_new:.4f} "
                    f"V[mid]={V[self.M // 2, self.L // 2]:.4f}"
                )

            V = self.step(V, tau_old, tau_new, theta=theta)

        return V, self.S, self.v


def price_extended_heston_option(
    S0,
    K,
    T,
    v0,
    kappa,
    theta_v,
    sigma,
    rho,
    r=0.0,
    q=0.0,
    M=120,
    L=80,
    N=120,
    S_max=None,
    v_max=None,
    option_type="call",
    theta_adi=0.5,
    verbose=False,
):
    """Price a European option under the extended Heston model.

    Parameters may be constants or functions of time-to-maturity tau.
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive")
    if K < 0:
        raise ValueError("K must be nonnegative")
    if T < 0:
        raise ValueError("T must be nonnegative")
    if v0 < 0:
        raise ValueError("v0 must be nonnegative")

    if S_max is None:
        S_max = 4.0 * max(S0, K, 1.0)

    if v_max is None:
        base_theta = _value(theta_v, 0.0)
        base_sigma = _value(sigma, 0.0)
        v_max = max(1.0, 5.0 * v0, 5.0 * base_theta, base_theta + 5.0 * base_sigma * np.sqrt(max(T, 0.0)))

    params = ExtendedHestonParams(
        kappa=kappa,
        theta=theta_v,
        sigma=sigma,
        rho=rho,
        v0=v0,
        r=r,
        q=q,
    )

    grid = Grid2DParams(S_max=S_max, v_max=v_max, M=M, L=L, N=N)
    solver = ExtendedHestonADI(params, grid, K=K, T=T, option_type=option_type)

    if verbose:
        print(f"Extended Heston ADI  K={K}  T={T}")
        print(f"  grid M={M}  L={L}  N={N}  S_max={S_max:.2f}  v_max={v_max:.4f}")

    V, S, v = solver.solve(theta=theta_adi, verbose=verbose)
    price = _bilinear(V, S, v, S0, v0)

    return price, V, S, v


def _bilinear(V, S, v, S0, v0):
    n_S, n_v = len(S), len(v)

    if S0 <= S[0]:
        i = 1
    elif S0 >= S[-1]:
        i = n_S - 1
    else:
        i = int(np.searchsorted(S, S0))
        if i == 0:
            i = 1

    if v0 <= v[0]:
        j = 1
    elif v0 >= v[-1]:
        j = n_v - 1
    else:
        j = int(np.searchsorted(v, v0))
        if j == 0:
            j = 1

    S_lo, S_hi = S[i - 1], S[i]
    v_lo, v_hi = v[j - 1], v[j]

    wS = float(np.clip((S0 - S_lo) / (S_hi - S_lo), 0.0, 1.0))
    wv = float(np.clip((v0 - v_lo) / (v_hi - v_lo), 0.0, 1.0))

    return (
        (1.0 - wS) * (1.0 - wv) * V[i - 1, j - 1]
        + wS * (1.0 - wv) * V[i, j - 1]
        + (1.0 - wS) * wv * V[i - 1, j]
        + wS * wv * V[i, j]
    )


def _thomas_batch(sub, mid, sup, rhs):
    """Vectorized Thomas algorithm for batched tridiagonal systems."""
    n = mid.shape[0]

    cp = np.empty_like(np.asarray(mid, dtype=float))
    dp = np.empty_like(rhs, dtype=float)

    cp[0] = sup[0] / mid[0]
    dp[0] = rhs[0] / mid[0]

    for i in range(1, n):
        denom = mid[i] - sub[i] * cp[i - 1]
        cp[i] = sup[i] / denom
        dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / denom

    x = np.empty_like(rhs, dtype=float)
    x[-1] = dp[-1]

    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x