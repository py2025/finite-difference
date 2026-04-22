from dataclasses import dataclass
import numpy as np
import math

@dataclass
class OptionParams:
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: str = "put"        # "call" or "put"
    exercise_type: str = "european" # "european" or "american"

@dataclass
class GridParams:
    S_max: float
    M: int  # stock steps
    N: int  # time steps

# Assertions
def validate_option_params(opt: OptionParams) -> None:
    if opt.S0 < 0:
        raise ValueError("S0: nonnegative")
    if opt.K <= 0:
        raise ValueError("K: positive")
    if opt.T < 0:
        raise ValueError("T: nonnegative")
    if opt.sigma < 0:
        raise ValueError("sigma: nonnegative")
    if opt.option_type not in ("call", "put"):
        raise ValueError("option_type: 'call' or 'put'")
    if opt.exercise_type not in ("european", "american"):
        raise ValueError("exercise: 'european' or 'american'")

def validate_grid_params(grid: GridParams) -> None:
    if grid.S_max <= 0:
        raise ValueError("S_max > 0")
    if grid.M < 2:
        raise ValueError("M >= 2")
    if grid.N < 1:
        raise ValueError("N >= 1")

# Grid Builder
def make_stock_grid(grid: GridParams) -> np.ndarray:
    return np.linspace(0.0, grid.S_max, grid.M + 1)

def ds(grid: GridParams) -> float:
    return grid.S_max / grid.M

def dt(opt: OptionParams, grid: GridParams) -> float:
    return opt.T / grid.N

# Value Functions
def payoff(S, K, option_type="put"):
    S = np.asarray(S)
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("option_type: 'call' or 'put'")

def intrinsic_value(S, opt: OptionParams):
    return payoff(S, opt.K, opt.option_type)

# Boundary Conditions
def boundary_values(t: float, opt: OptionParams, grid: GridParams):
    if not (0.0 <= t <= opt.T):
        raise ValueError("0 <= t <= T")

    tau = opt.T - t

    if opt.option_type == "call":
        left = 0.0
        right = grid.S_max - opt.K * math.exp(-opt.r * tau)
    else:
        if opt.exercise_type == "american":
            left = opt.K
        else:
            left = opt.K * math.exp(-opt.r * tau)
        right = 0.0

    return left, right

# Closed-form B-S Benchmark
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))

def bs_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    if S <= 0:
        return 0.0 if option_type == "call" else K * math.exp(-r * T)

    if sigma == 0:
        forward_payoff = max(S - K * math.exp(-r * T), 0.0)
        if option_type == "call":
            return forward_payoff
        return forward_payoff - S + K * math.exp(-r * T)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    elif option_type == "put":
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("option_type: 'call' or 'put'")

# Result Container
@dataclass
class FDResult:
    price: float
    S_grid: np.ndarray
    V: np.ndarray
    runtime: float | None = None

class BaseFDSolver:
    def __init__(self, opt: OptionParams, grid: GridParams):
        validate_option_params(opt)
        validate_grid_params(grid)
        self.opt = opt
        self.grid = grid
        self.S = make_stock_grid(grid)
        self.dS = ds(grid)
        self.dt = dt(opt, grid)

    def solve(self):
        raise NotImplementedError

