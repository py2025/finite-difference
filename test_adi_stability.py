#!/usr/bin/env python
"""
Quick test of ADI solver stability improvements
"""

import sys
sys.path.insert(0, '/root/project' if '/root' in sys.prefix else '.')

import numpy as np
import time
from src.core import SABRParams, Grid2DParams
from src.solvers.adi import ADISolver, price_sabr_option

print("=" * 70)
print("ADI SOLVER STABILITY TEST")
print("=" * 70)

# Test 1: Basic small grid
print("\nTest 1: Small grid (30, 20, 30)")
try:
    sabr = SABRParams(alpha=0.2, beta=1.0, rho=-0.5, nu=0.3)
    grid = Grid2DParams(S_max=300, v_max=0.6, M=30, L=20, N=30)
    solver = ADISolver(sabr=sabr, grid=grid, K=100, r=0.05, T=1.0, option_type="call")
    
    t_start = time.time()
    V, S, v = solver.solve(verbose=False)
    t_elapsed = time.time() - t_start
    
    # Check solution quality
    price_atm = V[len(S)//2, len(v)//2]
    is_valid = np.isfinite(price_atm) and 0 < price_atm < 100
    
    print(f"  Price at ATM: {price_atm:.6f}")
    print(f"  Time: {t_elapsed:.2f}s")
    print(f"  Status: {'✓ PASS' if is_valid else '✗ FAIL'}")
except Exception as e:
    print(f"  ✗ ERROR: {type(e).__name__}: {str(e)[:60]}")

# Test 2: Medium grid
print("\nTest 2: Medium grid (40, 25, 40)")
try:
    price, V, S, v = price_sabr_option(
        S0=100, K=100, T=1.0, r=0.05,
        alpha=0.2, beta=1.0, rho=-0.5, nu=0.3,
        M=40, L=25, N=40, verbose=False
    )
    
    is_valid = np.isfinite(price) and 0 < price < 100
    print(f"  Price: {price:.6f}")
    print(f"  Status: {'✓ PASS' if is_valid else '✗ FAIL'}")
except Exception as e:
    print(f"  ✗ ERROR: {type(e).__name__}: {str(e)[:60]}")

# Test 3: Larger grid
print("\nTest 3: Larger grid (50, 30, 50)")
try:
    price, V, S, v = price_sabr_option(
        S0=100, K=100, T=1.0, r=0.05,
        alpha=0.2, beta=1.0, rho=-0.5, nu=0.3,
        M=50, L=30, N=50, verbose=False
    )
    
    is_valid = np.isfinite(price) and 0 < price < 100
    print(f"  Price: {price:.6f}")
    print(f"  Status: {'✓ PASS' if is_valid else '✗ FAIL'}")
except Exception as e:
    print(f"  ✗ ERROR: {type(e).__name__}: {str(e)[:60]}")

print("\n" + "=" * 70)
print("Test complete. If all tests pass, the notebook should work!")
print("=" * 70)
