"""
Alternating Direction Implicit (ADI) method for 2D problems.
Solves SABR model option pricing PDE.

Reference: von Sydow et al. (2018) - Alternative Direction Implicit (ADI) method
"""

import numpy as np
from scipy.linalg import solve_banded
from src.core import SABRParams, Grid2DParams, make_stock_grid_2d, make_vol_grid_2d
from src.core import ds_2d, dv_2d, dt_2d


class ADISolver:
    """
    ADI solver for SABR model European option pricing.
    
    Solves the 2D PDE:
        ∂V/∂t + 1/2 * (α*S^β*v)^2 * ∂²V/∂S² + 1/2 * ν² * v² * ∂²V/∂v²
        + ρ * α * ν * S^β * v² * ∂²V/∂S∂v + r*S*∂V/∂S + (ν_drift)*∂V/∂v - r*V = 0
    
    with terminal condition V(S, v, T) = payoff(S, K, type)
    """
    
    def __init__(self, sabr: SABRParams, grid: Grid2DParams, K: float, r: float, 
                 T: float, option_type: str = "call"):
        """
        Parameters
        ----------
        sabr : SABRParams
            SABR model parameters (alpha, beta, rho, nu)
        grid : Grid2DParams
            2D grid specification (S_max, v_max, M, L, N)
        K : float
            Strike price
        r : float
            Risk-free rate
        T : float
            Time to maturity
        option_type : str
            "call" or "put"
        """
        self.sabr = sabr
        self.grid = grid
        self.K = K
        self.r = r
        self.T = T
        self.option_type = option_type
        
        # Grid setup
        self.S = make_stock_grid_2d(grid)
        self.v = make_vol_grid_2d(grid)
        self.dS = ds_2d(grid)
        self.dv = dv_2d(grid)
        self.dt = dt_2d(T, grid)
        
        self.M = grid.M
        self.L = grid.L
        self.N = grid.N
        
    def payoff(self, S):
        """Terminal payoff at maturity"""
        if self.option_type == "call":
            return np.maximum(S - self.K, 0.0)
        elif self.option_type == "put":
            return np.maximum(self.K - S, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def boundary_conditions(self, t, V):
        """Apply boundary conditions at S=0, S=S_max, v=0, v=v_max"""
        tau = self.T - t
        
        # S = 0 boundary
        if self.option_type == "call":
            V[0, :] = 0.0
        else:
            V[0, :] = self.K * np.exp(-self.r * tau)
        
        # S = S_max boundary
        if self.option_type == "call":
            V[self.M, :] = self.S[self.M] - self.K * np.exp(-self.r * tau)
        else:
            V[self.M, :] = 0.0
        
        # v = 0 boundary (zero volatility → deterministic dynamics)
        if self.option_type == "call":
            V[:, 0] = np.maximum(self.S * np.exp(self.r * tau) - self.K * np.exp(-self.r * tau), 0.0)
        else:
            V[:, 0] = np.maximum(self.K * np.exp(-self.r * tau) - self.S * np.exp(self.r * tau), 0.0)
        
        # v = v_max boundary (constant vol extrapolation)
        # Use forward difference approximation or constant extrapolation
        V[:, self.L] = V[:, self.L - 1]
        
        return V
    
    def coefficients_s(self, i, j):
        """
        Coefficients for S-direction sweep (direction 1)
        Using second-order finite differences
        """
        S_i = self.S[i]
        v_j = self.v[j]
        
        sigma_ij = self.sabr.alpha * (S_i ** self.sabr.beta) * v_j
        
        # S-direction coefficients
        a_coeff = (sigma_ij ** 2) / (2 * self.dS ** 2)  # ∂²V/∂S²
        b_coeff = (self.r * S_i) / (2 * self.dS)         # ∂V/∂S
        
        return a_coeff, b_coeff
    
    def coefficients_v(self, i, j):
        """
        Coefficients for v-direction sweep (direction 2)
        """
        v_j = self.v[j]
        
        # v-direction coefficients
        c_coeff = (self.sabr.nu ** 2 * v_j ** 2) / (2 * self.dv ** 2)  # ∂²V/∂v²
        
        return c_coeff
    
    def mixed_derivative_coefficient(self, i, j):
        """Coefficient for mixed derivative ∂²V/∂S∂v"""
        S_i = self.S[i]
        v_j = self.v[j]
        
        sigma_ij = self.sabr.alpha * (S_i ** self.sabr.beta) * v_j
        
        # Mixed derivative coefficient (discretized as central difference)
        rho_coeff = (self.sabr.rho * self.sabr.alpha * self.sabr.nu * 
                     (S_i ** self.sabr.beta) * v_j ** 2) / (4 * self.dS * self.dv)
        
        return rho_coeff
    
    def solve_step_s(self, V, theta=0.5):
        """
        ADI Step 1: Implicit in S-direction, explicit in v-direction
        theta = 0.5 for Crank-Nicolson splitting
        """
        M, L = self.M, self.L
        dt = self.dt
        
        # Storage for new values
        V_new = V.copy()
        
        # For each volatility level j, solve tridiagonal system in S
        for j in range(1, L):
            v_j = self.v[j]
            
            # Build tridiagonal system
            ab = np.zeros((3, M - 1))
            rhs = np.zeros(M - 1)
            
            for i in range(1, M):
                a, b = self.coefficients_s(i, j)
                c = self.coefficients_v(i, j)
                rho_coeff = self.mixed_derivative_coefficient(i, j)
                
                # Diagonal coefficient for S-sweep (implicit in S)
                diag_coeff = 1.0 + theta * dt * (2 * a + self.r)
                
                # Sub and superdiagonal coefficients
                lower_coeff = theta * dt * (-a + b)
                upper_coeff = theta * dt * (-a - b)
                
                if i == 1:
                    # First interior point
                    if i + 1 <= M - 1:
                        ab[0, i] = upper_coeff  # superdiagonal
                    ab[1, i - 1] = diag_coeff
                elif i == M - 1:
                    # Last interior point
                    if i - 1 >= 1:
                        ab[2, i - 2] = lower_coeff  # subdiagonal
                    ab[1, i - 1] = diag_coeff
                else:
                    # Interior points
                    ab[0, i] = upper_coeff      # superdiagonal
                    ab[1, i - 1] = diag_coeff
                    ab[2, i - 2] = lower_coeff  # subdiagonal
                
                # RHS: explicit part
                rhs[i - 1] = (
                    V[i, j]
                    + (1 - theta) * dt * (
                        -2 * a * V[i, j]
                        + a * (V[i + 1, j] + V[i - 1, j])
                        + b * (V[i + 1, j] - V[i - 1, j]) / 2
                        - c * (2 * V[i, j] - V[i, j + 1] - V[i, j - 1]) / self.dv ** 2
                        - rho_coeff * (V[i + 1, j + 1] - V[i + 1, j - 1] 
                                      - V[i - 1, j + 1] + V[i - 1, j - 1])
                        - self.r * V[i, j]
                    )
                )
            
            # Solve the tridiagonal system
            try:
                V_new[1:M, j] = solve_banded((1, 1), ab, rhs)
            except np.linalg.LinAlgError:
                # Fallback: use diagonal dominance approximation
                V_new[1:M, j] = rhs / np.diag(ab[1, :])
        
        return V_new
    
    def solve_step_v(self, V, theta=0.5):
        """
        ADI Step 2: Implicit in v-direction, explicit in S-direction
        """
        M, L = self.M, self.L
        dt = self.dt
        
        V_new = V.copy()
        
        # For each asset price level i, solve tridiagonal system in v
        for i in range(1, M):
            S_i = self.S[i]
            
            ab = np.zeros((3, L - 1))
            rhs = np.zeros(L - 1)
            
            for j in range(1, L):
                v_j = self.v[j]
                a, b = self.coefficients_s(i, j)
                c = self.coefficients_v(i, j)
                rho_coeff = self.mixed_derivative_coefficient(i, j)
                
                # Diagonal coefficient for v-sweep (implicit in v)
                diag_coeff = 1.0 + theta * dt * (2 * c + self.r)
                
                # Sub and superdiagonal coefficients
                lower_coeff = theta * dt * (-c)
                upper_coeff = theta * dt * (-c)
                
                if j == 1:
                    if j + 1 <= L - 1:
                        ab[0, j] = upper_coeff
                    ab[1, j - 1] = diag_coeff
                elif j == L - 1:
                    if j - 1 >= 1:
                        ab[2, j - 2] = lower_coeff
                    ab[1, j - 1] = diag_coeff
                else:
                    ab[0, j] = upper_coeff
                    ab[1, j - 1] = diag_coeff
                    ab[2, j - 2] = lower_coeff
                
                # RHS: explicit part
                rhs[j - 1] = (
                    V[i, j]
                    + (1 - theta) * dt * (
                        -2 * a * V[i, j]
                        + a * (V[i + 1, j] + V[i - 1, j])
                        + b * (V[i + 1, j] - V[i - 1, j]) / 2
                        - c * (V[i, j + 1] + V[i, j - 1] - 2 * V[i, j])
                        - rho_coeff * (V[i + 1, j + 1] - V[i + 1, j - 1]
                                      - V[i - 1, j + 1] + V[i - 1, j - 1])
                        - self.r * V[i, j]
                    )
                )
            
            try:
                V_new[i, 1:L] = solve_banded((1, 1), ab, rhs)
            except np.linalg.LinAlgError:
                V_new[i, 1:L] = rhs / np.diag(ab[1, :])
        
        return V_new
    
    def solve(self, theta=0.5, verbose=False):
        """
        Solve the SABR model PDE using ADI method
        
        Parameters
        ----------
        theta : float
            Weighting parameter (0.5 = Crank-Nicolson, 0 = fully explicit, 1 = fully implicit)
        verbose : bool
            Print progress
            
        Returns
        -------
        V : ndarray
            Option values at t=0 on the 2D grid (M+1, L+1)
        S : ndarray
            Asset price grid
        v : ndarray
            Volatility grid
        """
        # Initialize with terminal payoff
        V = np.zeros((self.M + 1, self.L + 1))
        for i in range(self.M + 1):
            V[i, :] = self.payoff(self.S[i])
        
        # Time stepping backward
        for n in range(self.N - 1, -1, -1):
            t_n = n * self.dt
            
            if verbose and n % max(1, self.N // 10) == 0:
                print(f"  Time step {self.N - n}/{self.N} (t={t_n:.4f})")
            
            # Apply boundary conditions
            V = self.boundary_conditions(t_n, V)
            
            # ADI Step 1: Implicit in S
            V = self.solve_step_s(V, theta=theta)
            
            # Apply boundary conditions again
            V = self.boundary_conditions(t_n, V)
            
            # ADI Step 2: Implicit in v
            V = self.solve_step_v(V, theta=theta)
            
            # Apply boundary conditions at end of step
            V = self.boundary_conditions(t_n, V)
            
            # Enforce non-negativity (options can't be negative)
            V = np.maximum(V, 0.0)
        
        return V, self.S, self.v


def price_sabr_option(S0: float, K: float, T: float, r: float,
                      alpha: float, beta: float, rho: float, nu: float,
                      M: int = 50, L: int = 30, N: int = 50,
                      v_max: float = None, option_type: str = "call",
                      S_max: float = None, verbose: bool = False):
    """
    High-level function to price a SABR option using ADI method.
    
    Parameters
    ----------
    S0 : float
        Initial asset price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    alpha : float
        SABR alpha parameter (volatility level)
    beta : float
        SABR beta parameter (0=normal, 1=lognormal)
    rho : float
        Correlation between asset and vol
    nu : float
        Volatility of volatility
    M : int
        Number of asset price steps
    L : int
        Number of volatility steps
    N : int
        Number of time steps
    v_max : float
        Maximum volatility (defaults to alpha * S0 * 3)
    option_type : str
        "call" or "put"
    S_max : float
        Maximum asset price (defaults to K * 3)
    verbose : bool
        Print progress
        
    Returns
    -------
    price : float
        Option price at (S0, alpha)
    V : ndarray
        Full price surface
    S : ndarray
        Asset price grid
    v : ndarray
        Volatility grid
    """
    if S_max is None:
        S_max = K * 3
    if v_max is None:
        v_max = alpha * S0 * 3
    
    sabr = SABRParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
    grid = Grid2DParams(S_max=S_max, v_max=v_max, M=M, L=L, N=N)
    
    solver = ADISolver(sabr=sabr, grid=grid, K=K, r=r, T=T, option_type=option_type)
    
    if verbose:
        print(f"Solving SABR option pricing with ADI method")
        print(f"  Parameters: K={K}, T={T}, r={r}")
        print(f"  SABR: alpha={alpha}, beta={beta}, rho={rho}, nu={nu}")
        print(f"  Grid: M={M}, L={L}, N={N}")
    
    V, S, v = solver.solve(verbose=verbose)
    
    # Interpolate price at initial condition S0, v0 = alpha
    v0 = alpha
    idx_S = np.searchsorted(S, S0)
    idx_v = np.searchsorted(v, v0)
    
    if idx_S >= len(S) or idx_v >= len(v):
        price = V[-1, -1]  # Fallback to boundary
    else:
        # Bilinear interpolation
        if idx_S > 0 and idx_S < len(S) - 1 and idx_v > 0 and idx_v < len(v) - 1:
            S_lo, S_hi = S[idx_S - 1], S[idx_S]
            v_lo, v_hi = v[idx_v - 1], v[idx_v]
            
            w_S = (S0 - S_lo) / (S_hi - S_lo)
            w_v = (v0 - v_lo) / (v_hi - v_lo)
            
            price = (
                (1 - w_S) * (1 - w_v) * V[idx_S - 1, idx_v - 1]
                + w_S * (1 - w_v) * V[idx_S, idx_v - 1]
                + (1 - w_S) * w_v * V[idx_S - 1, idx_v]
                + w_S * w_v * V[idx_S, idx_v]
            )
        else:
            price = V[idx_S, idx_v]
    
    return price, V, S, v
