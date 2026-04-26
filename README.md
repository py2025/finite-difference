# finite-difference

Finite-difference option pricing in 1-D (Black–Scholes) and 2-D (SABR stochastic volatility), with a Douglas-scheme ADI solver for the SABR PDE.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py2025/finite-difference/blob/master/notebooks/07_sabr_adi.ipynb)

## What it solves

European call and put options under either the Black–Scholes model (constant volatility) or the SABR model (stochastic volatility, with parameters α, β, ρ, ν). The Black–Scholes case is solved with explicit, implicit, and Crank–Nicolson schemes; the SABR case is a 2-D PDE solved by an Alternating Direction Implicit (ADI) method using Douglas operator splitting. The SABR solver reproduces a Black–Scholes price to machine-noise levels in the degenerate limit and converges at the expected second-order rate.

## Installation

```bash
git clone https://github.com/py2025/finite-difference.git
cd finite-difference
pip install -r requirements.txt
```

Requires Python 3.8+ with NumPy, SciPy, Matplotlib, and Jupyter (for notebooks).

## Quick start

```python
from src.solvers.adi import price_sabr_option

# SABR call: alpha=0.2, beta=1, rho=-0.3, nu=0.4
price, V, S, v = price_sabr_option(
    S0=100, K=100, T=1.0, r=0.05,
    alpha=0.2, beta=1.0, rho=-0.3, nu=0.4,
    option_type="call",
)
print(f"SABR call price: {price:.5f}")
```

```python
from src.solvers.crank_nicolson import solve_crank_nicolson

# 1-D Black-Scholes call via Crank-Nicolson
S, V = solve_crank_nicolson(
    S_max=300, K=100, r=0.05, sigma=0.2, T=1.0,
    M=200, N=200, option_type="call",
)
```

## API reference

### `price_sabr_option(...) -> (price, V, S, v)`

High-level SABR pricer. Builds the grid, runs the Douglas-ADI solver, and bilinearly interpolates the price at `(S0, alpha)`.

| Argument | Type | Description |
|---|---|---|
| `S0` | float | Initial asset price |
| `K` | float | Strike |
| `T` | float | Time to maturity (years) |
| `r` | float | Risk-free rate |
| `alpha` | float | SABR initial vol level (>0) |
| `beta` | float | SABR CEV exponent, in `[0, 1]` |
| `rho` | float | SABR correlation, in `[-1, 1]` |
| `nu` | float | SABR vol-of-vol (>0) |
| `M`, `L`, `N` | int | Grid points in S, v, time (default 80/40/80) |
| `S_max`, `v_max` | float, optional | Domain bounds (sensible defaults) |
| `option_type` | str | `"call"` or `"put"` |
| `theta` | float | ADI weight; 0.5 gives second-order accuracy |
| `verbose` | bool | Print per-step progress |

Returns `(price: float, V: ndarray (M+1, L+1), S: ndarray, v: ndarray)`.

### `ADISolver`

Lower-level access to the solver. `ADISolver(sabr, grid, K, r, T, option_type)` then `solver.solve(theta=0.5, verbose=False)` returns `(V, S, v)`. Useful when you want the full surface and not just the interpolated price at one point.

### 1-D Black–Scholes solvers

All in `src/solvers/`. Each takes `(S_max, K, r, sigma, T, M, N, option_type)` and returns `(S, V)` where `V` is the value at `t=0` on the asset grid.

| Function | Module | Notes |
|---|---|---|
| `solve_explicit` | `explicit` | CFL-limited; fastest per step |
| `solve_implicit` | `implicit` | Unconditionally stable, O(Δt) |
| `solve_crank_nicolson` | `crank_nicolson` | Unconditionally stable, O(Δt²) |
| `solve_american` | `american` | American put via projected CN |

### `bs_price(S, K, T, r, sigma, option_type) -> float`

Closed-form Black–Scholes price; used as the validation reference for the SABR solver in its degenerate limit.

## Demo notebook

The full SABR/ADI tutorial — derivation, solver, surface plots, convergence study, parity check, parameter sensitivity, and the implied-volatility smile — is in [`notebooks/07_sabr_adi.ipynb`](notebooks/07_sabr_adi.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/py2025/finite-difference/blob/master/notebooks/07_sabr_adi.ipynb)

## License

MIT.
