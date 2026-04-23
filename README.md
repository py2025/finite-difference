# Finite Difference Methods for Option Pricing

A comprehensive study of numerical methods for solving option pricing PDEs, from basic Black-Scholes to advanced stochastic volatility models.

## Project Evolution

### Phase 1: Black-Scholes Foundation (Original)
- Explicit finite difference method
- Implicit (backward Euler) method  
- Crank-Nicolson method
- Comparison of accuracy and runtime
- Reference: Notebooks 01-06

### Phase 2: Advanced SABR Model with ADI (New)
- **Alternative Direction Implicit (ADI)** method
- **SABR model** (Stochastic Alpha-Beta-Rho) 
- 2D PDE solving for stochastic volatility
- High accuracy with computational efficiency
- Reference: **Notebook 07** + **ADI_IMPLEMENTATION_GUIDE.md**

## Quick Start

### Basic Black-Scholes Pricing
```python
from src.solvers.crank_nicolson import solve_crank_nicolson

S, V = solve_crank_nicolson(
    S_max=300, K=100, r=0.05, sigma=0.2, T=1.0,
    M=100, N=100, option_type="call"
)
print(f"ATM Call Price: {V[len(V)//2]:.4f}")
```

### SABR Model Pricing (NEW)
```python
from src.solvers.adi import price_sabr_option

# Price with stochastic volatility
price, V, S, v = price_sabr_option(
    S0=100, K=100, T=1.0, r=0.05,
    alpha=0.2, beta=1.0, rho=-0.5, nu=0.3
)
print(f"SABR Call Price: {price:.4f}")
```

## File Structure

```
finite-difference/
├── README.md                          # This file
├── ADI_IMPLEMENTATION_GUIDE.md         # Detailed ADI documentation
├── requirements.txt                   # Python dependencies
│
├── src/
│   ├── core.py                        # Core data structures & utilities
│   │   ├── OptionParams, GridParams   # 1D parameters
│   │   ├── SABRParams, Grid2DParams   # 2D parameters (NEW)
│   │   ├── Black-Scholes pricing
│   │   └── Boundary condition helpers
│   │
│   └── solvers/
│       ├── explicit.py                # Explicit FD method
│       ├── implicit.py                # Implicit (backward Euler) method
│       ├── crank_nicolson.py          # Crank-Nicolson method
│       └── adi.py                     # ADI method for SABR (NEW)
│
└── notebooks/
    ├── 01_black_scholes_formula.ipynb       # Analytical baseline
    ├── 02_grid_and_boundary_conditions.ipynb # FD fundamentals
    ├── 03_european_implicit.ipynb            # Implicit method
    ├── 04_european_crank_nicolson.ipynb      # CN method
    ├── 05_european_explicit.ipynb            # Explicit method
    ├── 06_accuracy_runtime_comparison.ipynb  # Method comparison
    └── 07_sabr_adi.ipynb                     # ADI + SABR tutorial (NEW)
```

## Key Notebooks

### 07_sabr_adi.ipynb (NEW)
**The Von Sydow Paper Implementation**

Comprehensive tutorial covering:
1. SABR model parameters (α, β, ρ, ν)
2. 2D PDE formulation
3. ADI algorithm explanation
4. Boundary conditions
5. Full solver implementation
6. Visualizations (price surface, sensitivity)
7. Convergence analysis
8. Validation against Black-Scholes

**Run time**: ~10 minutes on standard hardware

### 06_accuracy_runtime_comparison.ipynb
Compares three 1D methods:
- Explicit (fast, limited stability)
- Implicit (stable, higher cost)
- Crank-Nicolson (optimal balance)

## Method Comparison

| Method | Dimension | Stability | Accuracy | Speed |
|--------|-----------|-----------|----------|-------|
| **Explicit** | 1D | CFL limited | O(Δx², Δt) | Very fast |
| **Implicit** | 1D | Unconditional | O(Δx², Δt²) | Medium |
| **Crank-Nicolson** | 1D | Unconditional | O(Δx², Δt²) | Medium |
| **ADI** | 2D | Unconditional | O(Δx², Δt²) | Efficient |

## Installation

```bash
# Clone repository
git clone <repo>
cd finite-difference

# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook
```

## Requirements

- Python 3.8+
- NumPy (numerical computing)
- SciPy (scientific algorithms)
- Matplotlib (visualization)
- Jupyter (interactive notebooks)

See `requirements.txt` for versions.

## Model Summary

### Black-Scholes (1973)
- 1D, constant volatility
- Closed-form solution exists
- Basis for 1D finite difference methods
- Limitations: assumes constant vol

### SABR Model (2002, Hagan et al.)
- 2D, stochastic volatility
- Fits market volatility smile well
- No closed-form solution
- Requires numerical methods (like ADI)
- α, β, ρ, ν parameters

## Mathematical Foundation

### Black-Scholes PDE
```
∂V/∂t + ½σ² S² ∂²V/∂S² + rS ∂V/∂S - rV = 0
```
Terminal condition: V(S,T) = max(S-K, 0) for call

### SABR PDE (2D)
```
∂V/∂t + ½(αS^β v)² ∂²V/∂S² + ½ν²v² ∂²V/∂v²
  + ραν S^β v² ∂²V/∂S∂v + rS ∂V/∂S - rV = 0
```
Terminal condition: V(S,v,T) = max(S-K, 0) for call

## Research Applications

✓ Option pricing in equity markets  
✓ Interest rate derivatives (β=0)  
✓ FX options (β=0.5)  
✓ Volatility smile modeling  
✓ Risk management (Greeks: δ, γ, ν)  
✓ Stochastic volatility calibration  

## Learning Outcomes

After working through this project, you will understand:

1. **Black-Scholes Model**
   - Closed-form solution
   - Limitations (constant vol)

2. **Finite Difference Methods**
   - Explicit, implicit, Crank-Nicolson
   - Stability analysis
   - Convergence properties

3. **SABR Model**
   - Stochastic volatility dynamics
   - Volatility smile
   - Parameter interpretation

4. **ADI Method**
   - Dimension splitting
   - Alternating sweeps
   - Computational efficiency
   - Unconditional stability

5. **Numerical Implementation**
   - Grid generation
   - Boundary conditions
   - Tridiagonal solvers
   - Error analysis

## References

### Foundational
1. Black, F., & Scholes, M. (1973). "The pricing of options and corporate liabilities." *Journal of Political Economy*, 81(3), 637-654.
2. Hagan, P., Kumar, D., Lesniewski, A., & Woodward, D. (2002). "Managing smile risk." *Wilmott Magazine*, 84-108.

### Numerical Methods
3. Rannacher, R., & Turek, S. (1992). "Simple nonconforming quadrilateral Stokes element." *Numerical Methods for Partial Differential Equations*, 8(2), 97-111.
4. Wilmott, P., Howison, S., & Dewynne, J. (1995). *The Mathematics of Financial Derivatives*. Cambridge University Press.

### Von Sydow Paper (PRIMARY REFERENCE)
5. **von Sydow, L., Lindström, E., Sherve, C., Wiktorsson, M., & Löfdahl, B. (2018).** 
   *"Options pricing using Alternative Direction Implicit (ADI) method."*
   **International Journal of Computer Mathematics**, 95(11), 2275-2286.
   https://doi.org/10.1080/00207160.2018.1544368

## Course Context

**Course**: MATH 5030 (Finite Difference Methods)  
**Instructor**: Jae  
**Group**: Group 16  
**Timeline**: 
- Phase 1: Black-Scholes foundation
- Phase 2: SABR + ADI (current)

## Future Extensions

- American option pricing with ADI
- Jump-diffusion models
- Multiple-asset derivatives
- GPU acceleration
- Market data calibration
- Greeks computation

## Contributing

This is a course project. For improvements or issues:
1. Document the concern clearly
2. Reference specific notebook/cell
3. Propose a solution
4. Test on provided examples

## License

Course project - MIT License

---

**Last Updated**: April 2026  
**Status**: Phase 2 Complete (ADI + SABR)
