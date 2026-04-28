"""
Alternating Direction Implicit (ADI) method for the SABR model.

Solves the 2D PDE for V(S, v, t) on a uniform Cartesian grid using the
Douglas operator-splitting scheme:

    F0 V = rho * nu * v^2 * S^beta * d^2 V/(dS dv)        (mixed; explicit)
    F1 V = (1/2) v^2 S^(2 beta) d^2V/dS^2 + r S dV/dS - (r/2) V   (implicit)
    F2 V = (1/2) nu^2 v^2 d^2V/dv^2 - (r/2) V             (implicit)

In time-to-maturity tau = T - t, the PDE is dV/dtau = (F0 + F1 + F2) V.
One Douglas step from tau_n -> tau_{n+1} (dt = tau_{n+1} - tau_n):

    Y0 = V^n + dt (F0 + F1 + F2) V^n                      (predictor)
    (I - theta dt F1) Y1 = Y0 - theta dt F1 V^n           (S-implicit)
    (I - theta dt F2) V^{n+1} = Y1 - theta dt F2 V^n      (v-implicit)

theta = 1/2 gives second-order accuracy (Crank-Nicolson-like). The
mixed-derivative operator is handled explicitly in the predictor only.

Reference: K. in 't Hout & S. Foulon (2010), "ADI finite difference schemes
for option pricing in the Heston model with correlation".
"""

import numpy as np

from src.core import (
    SABRParams,
    Grid2DParams,
    make_stock_grid_2d,
    make_vol_grid_2d,
    ds_2d,
    dv_2d,
    dt_2d,
)


class ADISolver:
    """Douglas-scheme ADI solver for European SABR options."""

    def __init__(
        self,
        sabr: SABRParams,
        grid: Grid2DParams,
        K: float,
        r: float,
        T: float,
        option_type: str = "call",
    ):
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")

        self.sabr = sabr
        self.grid = grid
        self.K = K
        self.r = r
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

        self._build_coefficients()

    def _build_coefficients(self):
        beta = self.sabr.beta
        nu = self.sabr.nu
        rho = self.sabr.rho

        S_col = self.S[:, None]   # (M+1, 1)
        v_row = self.v[None, :]   # (1, L+1)

        # F1 (S-direction)
        self.aS = 0.5 * (v_row ** 2) * (S_col ** (2.0 * beta))   # (M+1, L+1)
        self.bS = self.r * self.S                                # (M+1,)

        # F2 (v-direction)
        self.aV = 0.5 * (nu ** 2) * (self.v ** 2)                # (L+1,)

        # F0 (mixed)
        self.gamma = rho * nu * (v_row ** 2) * (S_col ** beta)   # (M+1, L+1)

    def payoff(self, S):
        S = np.asarray(S, dtype=float)
        if self.option_type == "call":
            return np.maximum(S - self.K, 0.0)
        return np.maximum(self.K - S, 0.0)

    def boundary_conditions(self, t, V):
        """Apply Dirichlet/Neumann BCs at calendar time t."""
        return self._apply_bc_tau(V, self.T - t)

    def _apply_bc_tau(self, V, tau):
        df = np.exp(-self.r * tau)
        S = self.S

        if self.option_type == "call":
            # Absorbing at S=0; deterministic asymptote at S_max
            V[0, :] = 0.0
            V[self.M, :] = self.S[self.M] - self.K * df
            # At v=0 the SABR vol is frozen at 0; payoff propagates deterministically
            V[:, 0] = np.maximum(S - self.K * df, 0.0)
        else:
            V[0, :] = self.K * df
            V[self.M, :] = 0.0
            V[:, 0] = np.maximum(self.K * df - S, 0.0)

        # Neumann at v_max:  dV/dv = 0  =>  V[:, L] = V[:, L-1]
        V[:, self.L] = V[:, self.L - 1]
        return V

    # ------------------------------------------------------------------
    # Spatial operators (applied to V; result is 0 on boundaries that the
    # operator does not touch, which is fine because we always re-apply BCs)
    # ------------------------------------------------------------------

    def _F0(self, V):
        out = np.zeros_like(V)
        out[1:-1, 1:-1] = (
            self.gamma[1:-1, 1:-1]
            * (V[2:, 2:] - V[2:, :-2] - V[:-2, 2:] + V[:-2, :-2])
            / (4.0 * self.dS * self.dv)
        )
        return out

    def _F1(self, V):
        out = np.zeros_like(V)
        a = self.aS[1:-1, :] / (self.dS ** 2)
        b = self.bS[1:-1, None] / (2.0 * self.dS)
        out[1:-1, :] = (
            a * (V[2:, :] - 2.0 * V[1:-1, :] + V[:-2, :])
            + b * (V[2:, :] - V[:-2, :])
            - 0.5 * self.r * V[1:-1, :]
        )
        return out

    def _F2(self, V):
        out = np.zeros_like(V)
        c = self.aV[None, 1:-1] / (self.dv ** 2)
        out[:, 1:-1] = (
            c * (V[:, 2:] - 2.0 * V[:, 1:-1] + V[:, :-2])
            - 0.5 * self.r * V[:, 1:-1]
        )
        return out

    # ------------------------------------------------------------------
    # Implicit sweeps
    # ------------------------------------------------------------------

    def _solve_S_implicit(self, rhs, theta_dt, tau_new):
        """Solve (I - theta_dt F1) Y = rhs for interior i, all interior j."""
        M, L = self.M, self.L
        Y = self._apply_bc_tau(rhs.copy(), tau_new)

        a = self.aS[1:M, 1:L] / (self.dS ** 2)         # (M-1, L-1)
        b = self.bS[1:M, None] / (2.0 * self.dS)        # (M-1, 1) -> broadcasts

        sub = -theta_dt * (a - b)                        # multiplies Y[i-1,j]
        mid = 1.0 + theta_dt * (2.0 * a + 0.5 * self.r)  # multiplies Y[i,  j]
        sup = -theta_dt * (a + b)                        # multiplies Y[i+1,j]

        rhs_int = rhs[1:M, 1:L].copy()
        # Move known boundary values to the RHS
        rhs_int[0, :]  -= sub[0, :]  * Y[0, 1:L]
        rhs_int[-1, :] -= sup[-1, :] * Y[M, 1:L]

        Y[1:M, 1:L] = _thomas_batch(sub, mid, sup, rhs_int)
        return self._apply_bc_tau(Y, tau_new)

    def _solve_v_implicit(self, rhs, theta_dt, tau_new):
        """Solve (I - theta_dt F2) Y = rhs for interior j, all i."""
        M, L = self.M, self.L
        Y = self._apply_bc_tau(rhs.copy(), tau_new)

        c = self.aV[1:L] / (self.dv ** 2)                # (L-1,)

        sub = -theta_dt * c
        mid = 1.0 + theta_dt * (2.0 * c + 0.5 * self.r)
        sup = -theta_dt * c

        # Neumann at v_max: Y[i, L] = Y[i, L-1] folds the last super-diagonal
        # entry into the diagonal at j = L-1.
        mid_eff = mid.copy()
        mid_eff[-1] = mid[-1] + sup[-1]

        # Solve column-wise across all i.  Transpose so the v-axis comes first.
        rhs_int = rhs[:, 1:L].T.copy()                   # (L-1, M+1)
        rhs_int[0, :] -= sub[0] * Y[:, 0]                # Dirichlet at v=0

        sol = _thomas_batch(sub, mid_eff, sup, rhs_int)  # (L-1, M+1)
        Y[:, 1:L] = sol.T
        Y[:, L] = Y[:, L - 1]                            # Neumann mirror
        return self._apply_bc_tau(Y, tau_new)

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def step(self, V, tau_old, tau_new, theta=0.5):
        # AGT: theta is the Douglas/CN weight; outside [0, 1] the sweeps silently overflow
        if not 0.0 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0, 1], got {theta}")
        dt = tau_new - tau_old

        F0V = self._F0(V)
        F1V = self._F1(V)
        F2V = self._F2(V)

        Y0 = V + dt * (F0V + F1V + F2V)
        Y0 = self._apply_bc_tau(Y0, tau_new)

        Y1 = self._solve_S_implicit(Y0 - theta * dt * F1V, theta * dt, tau_new)
        V_new = self._solve_v_implicit(Y1 - theta * dt * F2V, theta * dt, tau_new)
        return V_new

    def solve(self, theta=0.5, verbose=False):
        # Terminal condition (tau = 0)
        V = np.empty((self.M + 1, self.L + 1))
        V[:, :] = self.payoff(self.S)[:, None]
        V = self._apply_bc_tau(V, 0.0)

        for n in range(self.N):
            tau_new = (n + 1) * self.dt
            tau_old = n * self.dt
            if verbose and (n % max(1, self.N // 10) == 0):
                # AGT: surface a mid-grid value so divergence shows up in the log
                print(f"  step {n + 1}/{self.N}  tau={tau_new:.4f}  V[mid]={V[self.M // 2, self.L // 2]:.4f}")
            V = self.step(V, tau_old, tau_new, theta=theta)

        return V, self.S, self.v


# ----------------------------------------------------------------------
# Vectorised Thomas algorithm
# ----------------------------------------------------------------------

def _thomas_batch(sub, mid, sup, rhs):
    """
    Solve a batch of tridiagonal systems by the Thomas algorithm.

    Coefficients (sub, mid, sup) may be 1-D of length n (broadcast across
    every column of rhs) or 2-D matching rhs's shape (n, batch).  rhs is
    indexed (row, batch); the result has the same shape.

    Equation at row i:   sub[i] x[i-1] + mid[i] x[i] + sup[i] x[i+1] = rhs[i]
    sub[0] and sup[-1] are unused.
    """
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


# ----------------------------------------------------------------------
# High-level pricing API
# ----------------------------------------------------------------------

def price_sabr_option(
    S0,
    K,
    T,
    r,
    alpha,
    beta,
    rho,
    nu,
    M=80,
    L=40,
    N=80,
    v_max=None,
    S_max=None,
    option_type="call",
    theta=0.5,
    verbose=False,
):
    """Price a European SABR option via the Douglas ADI solver.

    Returns
    -------
    price : float
        Option value at (S0, alpha) interpolated bilinearly on the grid.
    V : ndarray, shape (M+1, L+1)
        Full price surface at t = 0.
    S : ndarray, shape (M+1,)
        Underlying-price grid.
    v : ndarray, shape (L+1,)
        Volatility grid.
    """
    if S_max is None:
        S_max = max(4.0 * K, 4.0 * S0)
    if v_max is None:
        v_max = max(5.0 * alpha, 1.0)
    if alpha > v_max:
        raise ValueError(f"alpha={alpha} exceeds v_max={v_max}")

    sabr = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
    grid = Grid2DParams(S_max=S_max, v_max=v_max, M=M, L=L, N=N)
    solver = ADISolver(sabr, grid, K=K, r=r, T=T, option_type=option_type)

    if verbose:
        print(f"ADI/SABR  K={K}  T={T}  r={r}")
        print(f"  alpha={alpha}  beta={beta}  rho={rho}  nu={nu}")
        print(f"  grid M={M}  L={L}  N={N}  S_max={S_max:.2f}  v_max={v_max:.2f}")

    V, S, v = solver.solve(theta=theta, verbose=verbose)
    price = _bilinear(V, S, v, S0, alpha)
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
