"""Shared fixtures used across the test suite."""

import pytest


@pytest.fixture
def atm():
    """Vanilla at-the-money parameters used as the default Black-Scholes
    setup across most tests."""
    return dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


@pytest.fixture
def sabr_params():
    """A typical SABR parameter set (alpha, beta, rho, nu)."""
    return dict(alpha=0.20, beta=1.0, rho=-0.3, nu=0.4)


@pytest.fixture(params=[
    (100, 100, 1.0, 0.05, 0.20),    # ATM, baseline
    (80, 100, 0.5, 0.03, 0.30),     # OTM call / ITM put
    (120, 100, 2.0, 0.07, 0.15),    # ITM call, long-dated
    (100, 110, 0.25, 0.00, 0.25),   # zero rate, short-dated
])
def parity_case(request):
    """A diverse sweep of (S, K, T, r, sigma) for parity-style tests
    that need to verify a property holds across regimes, not just ATM."""
    S, K, T, r, sigma = request.param
    return dict(S=S, K=K, T=T, r=r, sigma=sigma)
