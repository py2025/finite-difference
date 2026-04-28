# Quick Start Guide - ADI Method for SABR

## For Group 16: Professor's Request Implementation

Your professor asked to implement ADI method for the SABR model based on the von Sydow paper. This is now complete!

## What's New

✅ **ADI Solver** - Efficient 2D PDE solver  
✅ **SABR Model** - Stochastic volatility dynamics  
✅ **Tutorial Notebook** - `notebooks/07_sabr_adi.ipynb`  
✅ **Documentation** - `ADI_IMPLEMENTATION_GUIDE.md`  

## Run the Tutorial (5 minutes)

```bash
# 1. Navigate to project
cd c:\Users\py202\Documents\School\wn2026\MATH 5030\finite-difference

# 2. Start Jupyter
jupyter notebook

# 3. Open: notebooks/07_sabr_adi.ipynb
# 4. Run all cells (Ctrl+A, then Shift+Enter)
```

## Example: Price a SABR Option

```python
from finite_difference import price_sabr_option

# Example parameters (typical for equity options)
price, V, S, v = price_sabr_option(
    S0=100.0,      # Initial stock price
    K=100.0,       # Strike price
    T=1.0,         # 1 year to maturity
    r=0.05,        # 5% risk-free rate
    
    # SABR parameters
    alpha=0.2,     # Volatility level
    beta=1.0,      # 1.0 = lognormal (like Black-Scholes)
    rho=-0.5,      # -50% correlation
    nu=0.3,        # 30% vol-of-vol
)

print(f"Call price: ${price:.2f}")
```

## Key Parameter Ranges

### SABR Parameters (β in [0, 1])

| β | Model | Use Case |
|---|-------|----------|
| 0.0 | Normal | Interest rates, FX |
| 0.5 | CEV | Commodity futures |
| 1.0 | Lognormal | Equities (Black-Scholes limit) |

### Typical Values

```python
# Equity options
alpha = 0.15-0.25    # Vol level
beta = 1.0           # Lognormal
rho = -0.3 to -0.7   # Negative (smile)
nu = 0.2-0.5         # Vol-of-vol

# Interest rates
alpha = 0.01-0.03
beta = 0.0           # Normal
rho = -0.2 to 0.2
nu = 0.3-0.8
```

## Grid Resolution

Choose based on your needs:

```python
# Fast estimate (< 1 sec)
price_sabr_option(..., M=40, L=25, N=40)

# Good accuracy (2-5 sec)
price_sabr_option(..., M=60, L=40, N=60)  # DEFAULT

# High precision (10-30 sec)
price_sabr_option(..., M=100, L=60, N=100)
```

## Understanding the Output

```python
price, V, S, v = price_sabr_option(...)

# price: Single scalar, the option value at (S0, alpha)
print(price)  # e.g., 7.45

# V: Full price surface [M+1, L+1]
V.shape      # (61, 41)

# S: Asset price grid [0, S_max] with M+1 points
S[0]         # 0.0
S[-1]        # S_max

# v: Volatility grid [0, v_max] with L+1 points
v[0]         # 0.0  
v[-1]        # v_max
```

## Advanced Usage

### Direct Solver Interface

```python
from finite_difference import SABRParams, Grid2DParams, ADISolver

# Define parameters
sabr = SABRParams(
    alpha=0.2,
    beta=1.0,
    rho=-0.5,
    nu=0.3
)

# Define grid
grid = Grid2DParams(
    S_max=300.0,    # Asset price upper bound
    v_max=0.6,      # Volatility upper bound
    M=60,           # Asset price grid points
    L=40,           # Volatility grid points
    N=60            # Time steps
)

# Create solver
solver = ADISolver(
    sabr=sabr,
    grid=grid,
    K=100.0,        # Strike
    r=0.05,         # Rate
    T=1.0,          # Maturity
    option_type="call"
)

# Solve
V, S, v = solver.solve(verbose=True)

# Access specific values
atm_price = V[len(S)//2, len(v)//2]
```

### Plot Results

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

S_mesh, v_mesh = np.meshgrid(S, v)
ax.plot_surface(S_mesh.T, v_mesh.T, V)
ax.set_xlabel('Asset Price')
ax.set_ylabel('Volatility')
ax.set_zlabel('Option Value')
plt.show()

# 2D slice at fixed volatility
plt.plot(S, V[:, 20])
plt.xlabel('Asset Price')
plt.ylabel('Option Value')
plt.show()
```

## Verify Correctness

### Check Boundary Behavior

At strike K, option value should approximately equal stock value - discounted strike:
```python
idx_K = np.argmin(np.abs(S - K))
print(f"V at S=K: {V[idx_K, 10]:.4f}")  
# Should be positive for both calls and puts
```

### Check Monotonicity

Call prices should increase with asset price:
```python
call_prices = V[:, 20]
is_increasing = np.all(np.diff(call_prices) >= -1e-6)
print(f"Monotonicity check: {is_increasing}")
```

### Compare with Black-Scholes

For low vol-of-vol, SABR → Black-Scholes:
```python
from finite_difference import bs_price

# SABR with tiny vol-of-vol
price_sabr, _, _, _ = price_sabr_option(
    ..., nu=0.01  # Very small
)

# Black-Scholes
price_bs = bs_price(100, 100, 1.0, 0.05, 0.2, "call")

print(f"SABR: {price_sabr:.4f}")
print(f"B-S:  {price_bs:.4f}")
print(f"Diff: {abs(price_sabr - price_bs):.6f}")
```

## Troubleshooting

### ImportError: cannot import ADI solver
```python
# Make sure you're in the right directory
import sys
sys.path.insert(0, '../..')  # Adjust path
```

### Memory error on large grids
```python
# Use smaller grid
price_sabr_option(..., M=40, L=25, N=40)
```

### NaN values in output
```python
# Check parameter validity
# beta must be in [0, 1]
# rho must be in [-1, 1]
# alpha, nu must be > 0
```

## Next Steps for Your Group

1. **Understand the method** (15 min)
   - Read `ADI_IMPLEMENTATION_GUIDE.md`
   - Run notebook 07

2. **Verify accuracy** (30 min)
   - Run convergence analysis
   - Compare with Black-Scholes

3. **Explore parameters** (30 min)
   - Sensitivity analysis
   - Plot volatility smile

4. **Calibrate to market** (optional)
   - Collect real option prices
   - Calibrate α, ν, ρ
   - Price new options

5. **Compute Greeks** (optional extension)
   - Implement finite difference Greeks
   - Verify with analytical if available

## File Locations

```
📁 finite-difference/
  ├─ README.md (overview)
  ├─ ADI_IMPLEMENTATION_GUIDE.md (detailed)
  ├─ QUICK_START.md (this file)
  ├─ finite_difference/
  │   ├─ core.py (data structures)
  │   └─ solvers/adi.py (main algorithm)
  └─ notebooks/
      └─ 07_sabr_adi.ipynb (tutorial)
```

## Common Questions

**Q: Why ADI instead of just Crank-Nicolson?**  
A: CN would require solving large dense 2D systems. ADI splits it into 1D tridiagonal solves → much faster.

**Q: What does ρ (rho) control?**  
A: Correlation between asset and volatility. Negative ρ creates the "smile" in option prices.

**Q: Can I price American options?**  
A: This version is European. American pricing would require adding optimal stopping (future work).

**Q: How do I calibrate to market?**  
A: Minimize difference between model and market prices over α, ρ, ν (β usually fixed by market conventions).

## Success Metrics

✅ Notebook runs without errors  
✅ Prices are positive and reasonable  
✅ Convergence visible in analysis  
✅ Comparison with Black-Scholes reasonable  
✅ Sensitivity makes economic sense  

## Support

- **Documentation**: See `ADI_IMPLEMENTATION_GUIDE.md`
- **Code Comments**: Detailed docstrings in `src/solvers/adi.py`
- **Example Notebook**: `notebooks/07_sabr_adi.ipynb`

---

**Status**: ✅ Complete and tested  
**Last updated**: April 2026  
**Questions?** Check docstrings or notebook examples!
