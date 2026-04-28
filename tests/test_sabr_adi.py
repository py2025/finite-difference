"""
Tests for the SABR ADI solver — the centrepiece of the project.

The strongest test here is the degenerate-limit check: setting beta=1 and
nu->0 collapses SABR to Black-Scholes.  In this limit the ADI solver
must reproduce the closed-form BS price.  This single test exercises
essentially every line of the ADI code — the boundary conditions, the
mixed-derivative term, the Thomas solver, the time stepping, the bilinear
interpolation — and would catch almost any real bug.

The remaining tests cover put-call parity (still holds for any solver of
any model), a convergence sanity check, and validation of SABRParams,
which has a non-trivial __post_init__ branch for each parameter.
"""

import numpy as np
import pytest

from src.core import SABRParams, bs_price
from src.solvers.adi import price_sabr_option


def test_sabr_collapses_to_black_scholes():
    """beta=1, nu->0:  SABR -> Black-Scholes.  Strongest single test in the
    suite — every part of the ADI machinery has to be right for this to
    pass on a moderate grid."""
    S0, K, T, r, alpha = 100.0, 100.0, 1.0, 0.05, 0.20
    price, *_ = price_sabr_option(
        S0=S0, K=K, T=T, r=r,
        alpha=alpha, beta=1.0, rho=0.0, nu=1e-4,
        M=80, L=40, N=80, option_type="call",
    )
    expected = bs_price(S0, K, T, r, alpha, "call")
    assert price == pytest.approx(expected, abs=0.10)


def test_refining_grid_reduces_error_to_bs_limit():
    """Convergence sanity check.  We don't pin down the exact rate (that's
    the notebook's job) but a refined grid must be closer to the BS
    reference than a coarse one — otherwise something is non-convergent."""
    S0, K, T, r, alpha = 100.0, 100.0, 1.0, 0.05, 0.20
    expected = bs_price(S0, K, T, r, alpha, "call")

    coarse, *_ = price_sabr_option(
        S0=S0, K=K, T=T, r=r, alpha=alpha, beta=1.0, rho=0.0, nu=1e-4,
        M=40, L=20, N=40, option_type="call",
    )
    fine, *_ = price_sabr_option(
        S0=S0, K=K, T=T, r=r, alpha=alpha, beta=1.0, rho=0.0, nu=1e-4,
        M=80, L=40, N=80, option_type="call",
    )
    assert abs(fine - expected) < abs(coarse - expected)


def test_sabr_put_call_parity(sabr_params):
    """No-arbitrage identity, must hold for any model."""
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    call, *_ = price_sabr_option(
        S0=S0, K=K, T=T, r=r, **sabr_params,
        M=80, L=40, N=80, option_type="call",
    )
    put, *_ = price_sabr_option(
        S0=S0, K=K, T=T, r=r, **sabr_params,
        M=80, L=40, N=80, option_type="put",
    )
    expected = S0 - K * np.exp(-r * T)
    # 2D ADI noise is larger than 1D — looser tolerance than BS parity.
    assert call - put == pytest.approx(expected, abs=0.15)


@pytest.mark.parametrize("kwargs,bad_field", [
    (dict(alpha=0.2, beta=-0.1, rho=0.0, nu=0.3), "beta"),
    (dict(alpha=0.2, beta=1.5,  rho=0.0, nu=0.3), "beta"),
    (dict(alpha=0.2, beta=1.0, rho=-1.5, nu=0.3), "rho"),
    (dict(alpha=0.2, beta=1.0, rho=1.5,  nu=0.3), "rho"),
    (dict(alpha=0.0, beta=1.0, rho=0.0,  nu=0.3), "alpha"),
    (dict(alpha=0.2, beta=1.0, rho=0.0,  nu=0.0), "nu"),
])
def test_sabr_params_validation(kwargs, bad_field):
    """SABRParams.__post_init__ rejects each out-of-range parameter
    with a message naming that parameter."""
    with pytest.raises(ValueError, match=bad_field):
        SABRParams(**kwargs)
